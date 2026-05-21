//+------------------------------------------------------------------+
//|                                          stinger_fx_shim.mq5     |
//|                                          Stinger-Fx Shim EA      |
//+------------------------------------------------------------------+
//
// PURPOSE
//   This Expert Advisor is the in-MT5 half of the Stinger-Fx MT5 Strategy
//   Tester integration. The Python `MT5StrategyTester` spawns terminal64.exe
//   in tester mode with this EA attached; the EA forwards every tick to
//   Python over ZeroMQ REQ and applies the order action returned in REP.
//
// PROTOCOL (JSON over ZeroMQ, REQ from EA → REP from Python)
//   Tick request:
//     {"type":"tick","symbol":"EURUSD","time":1704067200,"bid":1.0823,"ask":1.0825}
//   Tick reply:
//     {"action":"NONE"}
//     {"action":"BUY","volume":0.01,"sl":1.0800,"tp":1.0850}
//     {"action":"SELL","volume":0.01}
//     {"action":"CLOSE","ticket":0}            (0 = close all)
//
// INSTALLATION (Windows MT5 only)
//   1. Copy libzmq-mt-4_3_5.dll (or your zmq build) into MT5/MQL5/Libraries/.
//   2. Copy this file into MT5/MQL5/Experts/StingerFx/.
//   3. In MetaEditor: open and compile → produces stinger_fx_shim.ex5.
//   4. In Strategy Tester: pick StingerFx/stinger_fx_shim, set inputs, run.
//
//+------------------------------------------------------------------+
#property copyright "Stinger-Fx"
#property version   "0.1"
#property strict

input string PythonEndpoint = "tcp://127.0.0.1:5555";   // Python REP socket
input int    RequestTimeoutMs = 5000;
input double DefaultVolume = 0.01;

// --- ZeroMQ DLL bindings (assumes libzmq is on Libraries path) -----------
#import "libzmq.dll"
   long zmq_ctx_new();
   int  zmq_ctx_term(long context);
   long zmq_socket(long context, int type);
   int  zmq_close(long socket);
   int  zmq_connect(long socket, uchar &endpoint[]);
   int  zmq_setsockopt(long socket, int option, uchar &value[], int value_size);
   int  zmq_send(long socket, uchar &buf[], int len, int flags);
   int  zmq_recv(long socket, uchar &buf[], int len, int flags);
#import

#define ZMQ_REQ 3
#define ZMQ_RCVTIMEO 27
#define ZMQ_LINGER 17

long g_ctx = 0;
long g_sock = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   g_ctx = zmq_ctx_new();
   g_sock = zmq_socket(g_ctx, ZMQ_REQ);

   uchar tmo[]; ArrayResize(tmo, 4);
   tmo[0] = (uchar)(RequestTimeoutMs & 0xFF);
   tmo[1] = (uchar)((RequestTimeoutMs >> 8) & 0xFF);
   tmo[2] = (uchar)((RequestTimeoutMs >> 16) & 0xFF);
   tmo[3] = (uchar)((RequestTimeoutMs >> 24) & 0xFF);
   zmq_setsockopt(g_sock, ZMQ_RCVTIMEO, tmo, 4);

   int linger = 0;
   uchar lng[]; ArrayResize(lng, 4);
   lng[0] = 0; lng[1] = 0; lng[2] = 0; lng[3] = 0;
   zmq_setsockopt(g_sock, ZMQ_LINGER, lng, 4);

   uchar ep[];
   StringToCharArray(PythonEndpoint, ep);
   zmq_connect(g_sock, ep);

   Print("stinger_fx_shim connected to ", PythonEndpoint);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_sock) zmq_close(g_sock);
   if(g_ctx)  zmq_ctx_term(g_ctx);
}

//+------------------------------------------------------------------+
void OnTick()
{
   MqlTick t;
   if(!SymbolInfoTick(_Symbol, t)) return;

   string req = StringFormat(
      "{\"type\":\"tick\",\"symbol\":\"%s\",\"time\":%d,\"bid\":%.5f,\"ask\":%.5f}",
      _Symbol, (long)t.time, t.bid, t.ask);

   uchar buf[];
   StringToCharArray(req, buf);
   if(zmq_send(g_sock, buf, ArraySize(buf) - 1, 0) < 0)
   {
      Print("zmq_send failed");
      return;
   }

   uchar reply[];
   ArrayResize(reply, 4096);
   int n = zmq_recv(g_sock, reply, ArraySize(reply), 0);
   if(n < 0)
   {
      Print("zmq_recv timeout/failed");
      return;
   }
   string s = CharArrayToString(reply, 0, n);
   ApplyAction(s);
}

//+------------------------------------------------------------------+
// Parse a tiny subset of JSON — enough for {"action":"BUY","volume":0.01,...}
string JsonField(const string body, const string key)
{
   int p = StringFind(body, "\"" + key + "\"");
   if(p < 0) return "";
   int colon = StringFind(body, ":", p);
   if(colon < 0) return "";
   int i = colon + 1;
   while(i < StringLen(body) && (StringSubstr(body, i, 1) == " " || StringSubstr(body, i, 1) == "\""))
      i++;
   int start = i;
   while(i < StringLen(body))
   {
      string c = StringSubstr(body, i, 1);
      if(c == "\"" || c == "," || c == "}") break;
      i++;
   }
   return StringSubstr(body, start, i - start);
}

//+------------------------------------------------------------------+
void ApplyAction(const string body)
{
   string action = JsonField(body, "action");
   if(action == "" || action == "NONE") return;

   double volume = StringToDouble(JsonField(body, "volume"));
   if(volume <= 0) volume = DefaultVolume;
   string sl_s = JsonField(body, "sl");
   string tp_s = JsonField(body, "tp");
   double sl = (sl_s == "" ? 0.0 : StringToDouble(sl_s));
   double tp = (tp_s == "" ? 0.0 : StringToDouble(tp_s));

   if(action == "BUY" || action == "SELL")
   {
      MqlTradeRequest req; ZeroMemory(req);
      MqlTradeResult  res; ZeroMemory(res);
      req.action = TRADE_ACTION_DEAL;
      req.symbol = _Symbol;
      req.volume = volume;
      req.type = (action == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      req.price = (action == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                    : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      req.sl = sl;
      req.tp = tp;
      req.deviation = 20;
      req.type_filling = ORDER_FILLING_IOC;
      OrderSend(req, res);
   }
   else if(action == "CLOSE")
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
         double vol = PositionGetDouble(POSITION_VOLUME);
         long type = PositionGetInteger(POSITION_TYPE);
         MqlTradeRequest req; ZeroMemory(req);
         MqlTradeResult  res; ZeroMemory(res);
         req.action = TRADE_ACTION_DEAL;
         req.position = ticket;
         req.symbol = _Symbol;
         req.volume = vol;
         req.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         req.price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                                 : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         req.deviation = 20;
         req.type_filling = ORDER_FILLING_IOC;
         OrderSend(req, res);
      }
   }
}
//+------------------------------------------------------------------+
