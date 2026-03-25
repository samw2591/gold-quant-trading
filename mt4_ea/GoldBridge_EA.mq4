//+------------------------------------------------------------------+
//| GoldBridge_EA.mq4 — 黄金量化交易桥接EA                           |
//| 功能: 接收Python指令，执行交易，回传状态                           |
//| 通信方式: 文件桥接 (MQL4/Files/DWX/)                              |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Trading System"
#property version   "1.00"
#property strict
#include <stdlib.mqh>  // 包含ErrorDescription函数

// 输入参数
extern int    TIMER_MS        = 500;     // 检查指令间隔(毫秒)
extern int    HEARTBEAT_SEC   = 5;       // 心跳间隔(秒)
extern string BRIDGE_SUBDIR   = "DWX";   // 桥接子目录

// 全局变量
string bridge_path;
datetime last_heartbeat;

//+------------------------------------------------------------------+
//| 初始化                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    bridge_path = BRIDGE_SUBDIR + "\\";
    
    // 创建桥接目录
    FolderCreate(bridge_path, 0);
    
    // 启动定时器
    EventSetMillisecondTimer(TIMER_MS);
    
    // 写入初始心跳
    WriteHeartbeat();
    
    // 写入账户信息
    WriteAccountInfo();
    
    // 写入持仓信息
    WritePositions();
    
    Print("[GoldBridge] EA 初始化完成. 桥接目录: ", bridge_path);
    Print("[GoldBridge] 等待 Python 指令...");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| 卸载                                                              |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("[GoldBridge] EA 已停止. 原因: ", reason);
}

//+------------------------------------------------------------------+
//| 定时器回调                                                        |
//+------------------------------------------------------------------+
void OnTimer()
{
    // 心跳
    if(TimeCurrent() - last_heartbeat >= HEARTBEAT_SEC)
    {
        WriteHeartbeat();
        WriteAccountInfo();
        WritePositions();
        last_heartbeat = TimeCurrent();
    }
    
    // 检查Python指令
    CheckCommands();
}

//+------------------------------------------------------------------+
//| 检查并执行Python指令                                              |
//+------------------------------------------------------------------+
void CheckCommands()
{
    string filename = bridge_path + "commands.json";
    
    if(!FileIsExist(filename, 0))
        return;
    
    // 读取指令文件
    int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
    if(handle == INVALID_HANDLE)
        return;
    
    string content = "";
    while(!FileIsEnding(handle))
        content += FileReadString(handle) + "\n";
    FileClose(handle);
    
    // 删除指令文件(防止重复执行)
    FileDelete(filename, 0);
    
    if(StringLen(content) < 5)
        return;
    
    Print("[GoldBridge] 收到指令: ", StringSubstr(content, 0, 100));
    
    // 解析JSON (简单解析)
    string action = ExtractJsonString(content, "action");
    
    if(action == "OPEN")
        ExecuteOpen(content);
    else if(action == "CLOSE")
        ExecuteClose(content);
    else if(action == "MODIFY")
        ExecuteModify(content);
    else
        WriteResponse(false, "未知操作: " + action);
}

//+------------------------------------------------------------------+
//| 执行开仓                                                          |
//+------------------------------------------------------------------+
void ExecuteOpen(string json)
{
    string symbol   = ExtractJsonString(json, "symbol");
    string type_str = ExtractJsonString(json, "type");
    double lots     = ExtractJsonDouble(json, "lots");
    double price    = ExtractJsonDouble(json, "price");
    double sl       = ExtractJsonDouble(json, "sl");
    double tp       = ExtractJsonDouble(json, "tp");
    string comment  = ExtractJsonString(json, "comment");
    int    magic    = (int)ExtractJsonDouble(json, "magic");
    int    slippage = (int)ExtractJsonDouble(json, "slippage");
    
    if(symbol == "") symbol = Symbol();
    if(slippage == 0) slippage = 5;
    
    int cmd = -1;
    if(type_str == "BUY")        cmd = OP_BUY;
    else if(type_str == "SELL")  cmd = OP_SELL;
    else if(type_str == "BUYLIMIT")  cmd = OP_BUYLIMIT;
    else if(type_str == "SELLLIMIT") cmd = OP_SELLLIMIT;
    else if(type_str == "BUYSTOP")   cmd = OP_BUYSTOP;
    else if(type_str == "SELLSTOP")  cmd = OP_SELLSTOP;
    
    if(cmd == -1)
    {
        WriteResponse(false, "无效订单类型: " + type_str);
        return;
    }
    
    // 市价单用当前价
    if(price == 0)
    {
        if(cmd == OP_BUY)  price = MarketInfo(symbol, MODE_ASK);
        if(cmd == OP_SELL) price = MarketInfo(symbol, MODE_BID);
    }
    
    // 打印详细参数用于调试
    Print("[GoldBridge] 下单参数: symbol=", symbol, " cmd=", cmd, " lots=", lots, 
          " price=", price, " sl=", sl, " tp=", tp, " slippage=", slippage, " magic=", magic);
    Print("[GoldBridge] 当前报价: Bid=", MarketInfo(symbol, MODE_BID), " Ask=", MarketInfo(symbol, MODE_ASK),
          " StopLevel=", MarketInfo(symbol, MODE_STOPLEVEL), " Digits=", (int)MarketInfo(symbol, MODE_DIGITS));
    
    // 检查最小止损距离
    double stopLevel = MarketInfo(symbol, MODE_STOPLEVEL) * MarketInfo(symbol, MODE_POINT);
    if(cmd == OP_BUY && sl > 0 && (price - sl) < stopLevel)
    {
        Print("[GoldBridge] ⚠️ 止损距离太近, 调整: ", (price - sl), " < ", stopLevel);
        sl = price - stopLevel - MarketInfo(symbol, MODE_POINT);
    }
    if(cmd == OP_SELL && sl > 0 && (sl - price) < stopLevel)
    {
        Print("[GoldBridge] ⚠️ 止损距离太近, 调整: ", (sl - price), " < ", stopLevel);
        sl = price + stopLevel + MarketInfo(symbol, MODE_POINT);
    }
    
    // 发送订单
    int ticket = OrderSend(symbol, cmd, lots, price, slippage, sl, tp, comment, magic, 0, clrGreen);
    
    if(ticket > 0)
    {
        Print("[GoldBridge] ✅ 开仓成功: #", ticket, " ", type_str, " ", symbol, " ", lots, "手 @ ", price, " SL=", sl);
        WriteResponse(true, "开仓成功 #" + IntegerToString(ticket));
        WritePositions();
    }
    else
    {
        int err = GetLastError();
        Print("[GoldBridge] ❌ 开仓失败: Error ", err, 
              " (", ErrorDescription(err), ")",
              " price=", price, " sl=", sl, " lots=", lots);
        WriteResponse(false, "开仓失败 Error: " + IntegerToString(err) + " " + ErrorDescription(err));
    }
}

//+------------------------------------------------------------------+
//| 执行平仓                                                          |
//+------------------------------------------------------------------+
void ExecuteClose(string json)
{
    int ticket = (int)ExtractJsonDouble(json, "ticket");
    
    if(!OrderSelect(ticket, SELECT_BY_TICKET))
    {
        WriteResponse(false, "找不到订单 #" + IntegerToString(ticket));
        return;
    }
    
    double price;
    if(OrderType() == OP_BUY)
        price = MarketInfo(OrderSymbol(), MODE_BID);
    else
        price = MarketInfo(OrderSymbol(), MODE_ASK);
    
    bool result = OrderClose(ticket, OrderLots(), price, 5, clrRed);
    
    if(result)
    {
        Print("[GoldBridge] ✅ 平仓成功: #", ticket);
        WriteResponse(true, "平仓成功 #" + IntegerToString(ticket));
        WritePositions();
    }
    else
    {
        int err = GetLastError();
        Print("[GoldBridge] ❌ 平仓失败: Error ", err);
        WriteResponse(false, "平仓失败 Error: " + IntegerToString(err));
    }
}

//+------------------------------------------------------------------+
//| 执行修改                                                          |
//+------------------------------------------------------------------+
void ExecuteModify(string json)
{
    int    ticket = (int)ExtractJsonDouble(json, "ticket");
    double sl     = ExtractJsonDouble(json, "sl");
    double tp     = ExtractJsonDouble(json, "tp");
    
    if(!OrderSelect(ticket, SELECT_BY_TICKET))
    {
        WriteResponse(false, "找不到订单 #" + IntegerToString(ticket));
        return;
    }
    
    if(sl == 0) sl = OrderStopLoss();
    if(tp == 0) tp = OrderTakeProfit();
    
    bool result = OrderModify(ticket, OrderOpenPrice(), sl, tp, 0, clrBlue);
    
    if(result)
    {
        WriteResponse(true, "修改成功 #" + IntegerToString(ticket));
        WritePositions();
    }
    else
    {
        int err = GetLastError();
        WriteResponse(false, "修改失败 Error: " + IntegerToString(err));
    }
}

//+------------------------------------------------------------------+
//| 写心跳文件                                                        |
//+------------------------------------------------------------------+
void WriteHeartbeat()
{
    string filename = bridge_path + "heartbeat.json";
    int handle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, "{\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + 
                        "\",\"symbol\":\"" + Symbol() +
                        "\",\"bid\":" + DoubleToString(Bid, Digits) +
                        ",\"ask\":" + DoubleToString(Ask, Digits) + "}");
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| 写账户信息                                                        |
//+------------------------------------------------------------------+
void WriteAccountInfo()
{
    string filename = bridge_path + "account.json";
    int handle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, 
            "{\"balance\":" + DoubleToString(AccountBalance(), 2) +
            ",\"equity\":" + DoubleToString(AccountEquity(), 2) +
            ",\"margin\":" + DoubleToString(AccountMargin(), 2) +
            ",\"free_margin\":" + DoubleToString(AccountFreeMargin(), 2) +
            ",\"leverage\":" + IntegerToString(AccountLeverage()) +
            ",\"bid\":" + DoubleToString(Bid, Digits) +
            ",\"ask\":" + DoubleToString(Ask, Digits) +
            ",\"spread\":" + IntegerToString(MarketInfo(Symbol(), MODE_SPREAD)) +
            ",\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"}");
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| 写持仓信息                                                        |
//+------------------------------------------------------------------+
void WritePositions()
{
    string filename = bridge_path + "positions.json";
    int handle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if(handle == INVALID_HANDLE) return;
    
    string json = "{\"positions\":[";
    bool first = true;
    
    for(int i = 0; i < OrdersTotal(); i++)
    {
        if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
        
        if(!first) json += ",";
        first = false;
        
        double current_price;
        if(OrderType() == OP_BUY)
            current_price = MarketInfo(OrderSymbol(), MODE_BID);
        else
            current_price = MarketInfo(OrderSymbol(), MODE_ASK);
        
        json += "{\"ticket\":" + IntegerToString(OrderTicket()) +
                ",\"symbol\":\"" + OrderSymbol() + "\"" +
                ",\"type\":" + IntegerToString(OrderType()) +
                ",\"lots\":" + DoubleToString(OrderLots(), 2) +
                ",\"open_price\":" + DoubleToString(OrderOpenPrice(), Digits) +
                ",\"current_price\":" + DoubleToString(current_price, Digits) +
                ",\"sl\":" + DoubleToString(OrderStopLoss(), Digits) +
                ",\"tp\":" + DoubleToString(OrderTakeProfit(), Digits) +
                ",\"profit\":" + DoubleToString(OrderProfit(), 2) +
                ",\"magic\":" + IntegerToString(OrderMagicNumber()) +
                ",\"comment\":\"" + OrderComment() + "\"" +
                ",\"open_time\":\"" + TimeToString(OrderOpenTime(), TIME_DATE|TIME_SECONDS) + "\"}";
    }
    
    json += "],\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"}";
    
    FileWriteString(handle, json);
    FileClose(handle);
}

//+------------------------------------------------------------------+
//| 写响应文件                                                        |
//+------------------------------------------------------------------+
void WriteResponse(bool success, string message)
{
    string filename = bridge_path + "response.json";
    int handle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
    if(handle != INVALID_HANDLE)
    {
        string result = success ? "true" : "false";
        FileWriteString(handle, 
            "{\"success\":" + result +
            ",\"message\":\"" + message + "\"" +
            ",\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"}");
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| 简单JSON解析                                                      |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
    string search = "\"" + key + "\":\"";
    int start = StringFind(json, search);
    if(start == -1) return "";
    start += StringLen(search);
    int end = StringFind(json, "\"", start);
    if(end == -1) return "";
    return StringSubstr(json, start, end - start);
}

double ExtractJsonDouble(string json, string key)
{
    string search1 = "\"" + key + "\":";
    int start = StringFind(json, search1);
    if(start == -1) return 0;
    start += StringLen(search1);
    
    // 跳过引号(如果值是字符串格式的数字)
    if(StringGetChar(json, start) == '"') start++;
    
    string num = "";
    for(int i = start; i < StringLen(json); i++)
    {
        int ch = StringGetChar(json, i);
        if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-')
            num += CharToString((uchar)ch);
        else
            break;
    }
    
    if(num == "") return 0;
    return StringToDouble(num);
}
//+------------------------------------------------------------------+
