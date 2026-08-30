import os
import logging
import asyncio

import pandas as pd
import yfinance as yf

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================================================
# GOLD PRO ANALYSIS BOT
# =========================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

SYMBOL = "GC=F"
TIMEFRAME = "15m"
DATA_PERIOD = "10d"


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================================================
# RSI
# =========================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# =========================================================
# MACD
# =========================================================

def calculate_macd(close):

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


# =========================================================
# ATR
# =========================================================

def calculate_atr(data, period=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        span=period,
        adjust=False
    ).mean()


# =========================================================
# GET GOLD DATA
# =========================================================

def get_gold_data():

    try:

        logger.info("Downloading gold data...")

        data = yf.download(
            SYMBOL,
            period=DATA_PERIOD,
            interval=TIMEFRAME,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if data is None or data.empty:

            logger.error("No data received.")

            return None

        # Handle Yahoo Finance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required:

            if column not in data.columns:

                logger.error(
                    "Missing column: %s",
                    column
                )

                return None

        data = data[required].copy()

        data = data.dropna()

        if len(data) < 250:

            logger.error(
                "Not enough candles: %s",
                len(data)
            )

            return None

        # =====================================================
        # EMA
        # =====================================================

        data["EMA20"] = data["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        data["EMA50"] = data["Close"].ewm(
            span=50,
            adjust=False
        ).mean()

        data["EMA200"] = data["Close"].ewm(
            span=200,
            adjust=False
        ).mean()

        # =====================================================
        # RSI
        # =====================================================

        data["RSI"] = calculate_rsi(
            data["Close"],
            14
        )

        # =====================================================
        # MACD
        # =====================================================

        (
            data["MACD"],
            data["MACD_SIGNAL"],
            data["MACD_HIST"]
        ) = calculate_macd(
            data["Close"]
        )

        # =====================================================
        # ATR
        # =====================================================

        data["ATR"] = calculate_atr(
            data,
            14
        )

        # =====================================================
        # SUPPORT / RESISTANCE
        # =====================================================

        data["SUPPORT"] = data["Low"].rolling(
            20
        ).min()

        data["RESISTANCE"] = data["High"].rolling(
            20
        ).max()

        # =====================================================
        # VOLUME
        # =====================================================

        data["VOLUME_AVG"] = data["Volume"].rolling(
            20
        ).mean()

        data = data.dropna()

        return data

    except Exception as e:

        logger.exception(
            "Data error: %s",
            e
        )

        return None


# =========================================================
# CANDLE ANALYSIS
# =========================================================

def candle_analysis(row):

    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])

    candle_range = high - low

    if candle_range <= 0:

        return 0, 0

    body = abs(close - open_price)

    upper_wick = high - max(
        open_price,
        close
    )

    lower_wick = min(
        open_price,
        close
    ) - low

    buy_score = 0
    sell_score = 0

    # Strong bullish candle
    if close > open_price:

        if body / candle_range >= 0.55:

            buy_score += 5

    # Strong bearish candle
    if close < open_price:

        if body / candle_range >= 0.55:

            sell_score += 5

    # Bullish rejection
    if lower_wick > body * 1.5:

        buy_score += 4

    # Bearish rejection
    if upper_wick > body * 1.5:

        sell_score += 4

    return buy_score, sell_score


# =========================================================
# MARKET ANALYSIS
# =========================================================

def analyze_market():

    data = get_gold_data()

    if data is None or data.empty:

        return None

    try:

        current = data.iloc[-1]
        previous = data.iloc[-2]

        price = float(current["Close"])

        ema20 = float(current["EMA20"])
        ema50 = float(current["EMA50"])
        ema200 = float(current["EMA200"])

        rsi = float(current["RSI"])

        macd = float(current["MACD"])
        macd_signal = float(
            current["MACD_SIGNAL"]
        )

        macd_hist = float(
            current["MACD_HIST"]
        )

        atr = float(current["ATR"])

        support = float(
            current["SUPPORT"]
        )

        resistance = float(
            current["RESISTANCE"]
        )

        volume = float(
            current["Volume"]
        )

        volume_avg = float(
            current["VOLUME_AVG"]
        )

        buy_score = 0
        sell_score = 0

        buy_reasons = []
        sell_reasons = []

        # =====================================================
        # TREND
        # =====================================================

        if (
            price > ema20
            and ema20 > ema50
            and ema50 > ema200
        ):

            buy_score += 20

            buy_reasons.append(
                "اتجاه صاعد قوي"
            )

        elif (
            price < ema20
            and ema20 < ema50
            and ema50 < ema200
        ):

            sell_score += 20

            sell_reasons.append(
                "اتجاه هابط قوي"
            )

        else:

            if price > ema20:

                buy_score += 7

            elif price < ema20:

                sell_score += 7

            if ema20 > ema50:

                buy_score += 7

            elif ema20 < ema50:

                sell_score += 7

        # =====================================================
        # RSI
        # =====================================================

        if 52 <= rsi <= 68:

            buy_score += 15

            buy_reasons.append(
                "RSI يدعم الصعود"
            )

        elif 32 <= rsi <= 48:

            sell_score += 15

            sell_reasons.append(
                "RSI يدعم الهبوط"
            )

        if rsi > 70:

            buy_score -= 10

        if rsi < 30:

            sell_score -= 10

        # =====================================================
        # MACD
        # =====================================================

        if (
            macd > macd_signal
            and macd_hist > 0
        ):

            buy_score += 15

            buy_reasons.append(
                "MACD صاعد"
            )

        elif (
            macd < macd_signal
            and macd_hist < 0
        ):

            sell_score += 15

            sell_reasons.append(
                "MACD هابط"
            )

        # =====================================================
        # MOMENTUM
        # =====================================================

        previous_close = float(
            previous["Close"]
        )

        if price > previous_close:

            buy_score += 8

        elif price < previous_close:

            sell_score += 8

        # =====================================================
        # SUPPORT / RESISTANCE
        # =====================================================

        support_distance = (
            price - support
        )

        resistance_distance = (
            resistance - price
        )

        if (
            support_distance >= 0
            and support_distance <= atr * 0.8
        ):

            buy_score += 8

            buy_reasons.append(
                "السعر قريب من الدعم"
            )

        if (
            resistance_distance >= 0
            and resistance_distance <= atr * 0.8
        ):

            sell_score += 8

            sell_reasons.append(
                "السعر قريب من المقاومة"
            )

        # =====================================================
        # VOLUME
        # =====================================================

        if volume_avg > 0:

            volume_ratio = (
                volume / volume_avg
            )

            if volume_ratio >= 1.2:

                if price > previous_close:

                    buy_score += 6

                    buy_reasons.append(
                        "حجم تداول مرتفع"
                    )

                elif price < previous_close:

                    sell_score += 6

                    sell_reasons.append(
                        "حجم تداول مرتفع"
                    )

        # =====================================================
        # CANDLE
        # =====================================================

        candle_buy, candle_sell = candle_analysis(
            current
        )

        buy_score += candle_buy
        sell_score += candle_sell

        # =====================================================
        # SCORE LIMIT
        # =====================================================

        buy_score = max(
            0,
            min(100, buy_score)
        )

        sell_score = max(
            0,
            min(100, sell_score)
        )

        # =====================================================
        # FINAL SIGNAL
        # =====================================================

        difference = abs(
            buy_score - sell_score
        )

        if (
            buy_score >= 60
            and buy_score > sell_score
            and difference >= 12
        ):

            signal = "🟢 BUY"
            confidence = buy_score
            direction = "BUY"

        elif (
            sell_score >= 60
            and sell_score > buy_score
            and difference >= 12
        ):

            signal = "🔴 SELL"
            confidence = sell_score
            direction = "SELL"

        else:

            signal = "🟡 WAIT"
            confidence = max(
                buy_score,
                sell_score
            )
            direction = "WAIT"

        # =====================================================
        # TRADE LEVELS
        # =====================================================

        entry = price

        if atr <= 0:

            atr = price * 0.001

        if direction == "BUY":

            stop_loss = entry - (
                atr * 1.5
            )

            risk = entry - stop_loss

            tp1 = entry + (
                risk * 1.5
            )

            tp2 = entry + (
                risk * 2.5
            )

        elif direction == "SELL":

            stop_loss = entry + (
                atr * 1.5
            )

            risk = stop_loss - entry

            tp1 = entry - (
                risk * 1.5
            )

            tp2 = entry - (
                risk * 2.5
            )

        else:

            stop_loss = None
            tp1 = None
            tp2 = None

        # =====================================================
        # RSI STATUS
        # =====================================================

        if rsi >= 70:

            rsi_status = "تشبع شراء"

        elif rsi <= 30:

            rsi_status = "تشبع بيع"

        elif rsi >= 50:

            rsi_status = "ميل صاعد"

        else:

            rsi_status = "ميل هابط"

        return {

            "price": price,

            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,

            "rsi": rsi,
            "rsi_status": rsi_status,

            "macd": macd,
            "macd_signal": macd_signal,

            "atr": atr,

            "support": support,
            "resistance": resistance,

            "buy_score": buy_score,
            "sell_score": sell_score,

            "signal": signal,
            "confidence": confidence,

            "entry": entry,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,

            "buy_reasons": buy_reasons,
            "sell_reasons": sell_reasons,
        }

    except Exception as e:

        logger.exception(
            "Analysis error: %s",
            e
        )

        return None


# =========================================================
# CREATE TELEGRAM MESSAGE
# =========================================================

def create_analysis_message():

    result = analyze_market()

    if result is None:

        return (
            "⚠️ تعذر الحصول على بيانات الذهب الآن.\n\n"
            "حاول مرة أخرى بعد قليل."
        )

    message = (
        "🪙 GOLD PRO ANALYSIS BOT\n"
        "\n"
        "━━━━━━━━━━━━━━━━\n"
        "\n"
        "📊 الذهب: XAUUSD\n"
        "⏱️ الإطار: M15\n"
        "\n"
        "🎯 الإشارة:\n"
        f"{result['signal']}\n"
        "\n"
        f"💪 قوة الإشارة:\n"
        f"{result['confidence']:.0f}%\n"
        "\n"
        "━━━━━━━━━━━━━━━━\n"
        "\n"
        f"💰 السعر:\n"
        f"{result['price']:.2f}\n"
        "\n"
        f"📈 EMA20:\n"
        f"{result['ema20']:.2f}\n"
        "\n"
        f"📊 EMA50:\n"
        f"{result['ema50']:.2f}\n"
        "\n"
        f"📉 EMA200:\n"
        f"{result['ema200']:.2f}\n"
        "\n"
        f"〽️ RSI(14):\n"
        f"{result['rsi']:.2f}\n"
        f"({result['rsi_status']})\n"
        "\n"
        f"📊 MACD:\n"
        f"{result['macd']:.4f}\n"
        "\n"
        f"🛡️ ATR:\n"
        f"{result['atr']:.2f}\n"
        "\n"
        f"🔵 الدعم:\n"
        f"{result['support']:.2f}\n"
        "\n"
        f"🔴 المقاومة:\n"
        f"{result['resistance']:.2f}\n"
        "\n"
        "━━━━━━━━━━━━━━━━\n"
        "\n"
        f"🟢 BUY SCORE: "
        f"{result['buy_score']}/100\n"
        f"🔴 SELL SCORE: "
        f"{result['sell_score']}/100\n"
    )

    # =====================================================
    # ENTRY / SL / TP
    # =====================================================

    if result["signal"] != "🟡 WAIT":

        message += (
            "\n"
            "━━━━━━━━━━━━━━━━\n"
            "\n"
            f"🎯 ENTRY:\n"
            f"{result['entry']:.2f}\n"
            "\n"
            f"🛑 STOP LOSS:\n"
            f"{result['stop_loss']:.2f}\n"
            "\n"
            f"🎯 TAKE PROFIT 1:\n"
            f"{result['tp1']:.2f}\n"
            "\n"
            f"🎯 TAKE PROFIT 2:\n"
            f"{result['tp2']:.2f}\n"
        )

    # =====================================================
    # REASONS
    # =====================================================

    if result["signal"] == "🟢 BUY":

        reasons = result["buy_reasons"]

    elif result["signal"] == "🔴 SELL":

        reasons = result["sell_reasons"]

    else:

        reasons = []

    if reasons:

        message += (
            "\n"
            "━━━━━━━━━━━━━━━━\n"
            "\n"
            "🔎 أسباب الإشارة:\n"
        )

        for reason in reasons[:5]:

            message += f"• {reason}\n"

    message += (
        "\n"
        "━━━━━━━━━━━━━━━━\n"
        "\n"
        "⚠️ تحليل فني فقط وليس ضمانًا للربح.\n"
        "اختبر الاستراتيجية على حساب تجريبي قبل استخدامها بأموال حقيقية."
    )

    return message


# =========================================================
# START COMMAND
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🪙 أهلاً بك في GOLD PRO ANALYSIS BOT!\n\n"
        "📊 تحليل XAUUSD على M15\n\n"
        "المؤشرات المستخدمة:\n"
        "• EMA20 / EMA50 / EMA200\n"
        "• RSI\n"
        "• MACD\n"
        "• ATR\n"
        "• Support / Resistance\n"
        "• Candle confirmation\n"
        "• Volume\n\n"
        "استخدم /gold للحصول على التحليل."
    )


# =========================================================
# TEST COMMAND
# =========================================================

async def test(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "✅ GOLD PRO BOT يعمل بنجاح!\n\n"
        "Telegram connection: OK"
    )


# =========================================================
# GOLD COMMAND
# =========================================================

async def gold(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "⏳ جاري تحليل الذهب باستخدام النظام المتقدم..."
    )

    message = await asyncio.to_thread(
        create_analysis_message
    )

    await update.message.reply_text(
        message
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN غير موجود في Railway Variables."
        )

    logger.info(
        "Starting GOLD PRO ANALYSIS BOT..."
    )

    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        CommandHandler(
            "test",
            test
        )
    
