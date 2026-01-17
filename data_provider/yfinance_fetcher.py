# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, RealtimeQuote

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现
    
    优先级：4（最低，作为兜底）
    数据来源：Yahoo Finance
    
    关键策略：
    - 自动转换股票代码格式
    - 处理时区和数据格式差异
    - 失败后指数退避重试
    
    注意事项：
    - A 股数据可能有延迟
    - 某些股票可能无数据
    - 数据精度可能与国内源略有差异
    """
    
    name = "YfinanceFetcher"
    priority = 4
    
    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式
        
        Yahoo Finance A 股代码格式：
        - 沪市：600519.SS (Shanghai Stock Exchange)
        - 深市：000001.SZ (Shenzhen Stock Exchange)
        - 美股：AAPL (直接使用)
        
        Args:
            stock_code: 原始代码，如 '600519', '000001', 'AAPL'
            
        Returns:
            Yahoo Finance 格式代码，如 '600519.SS', '000001.SZ', 'AAPL'
        """
        code = stock_code.strip()
        
        # 已经包含后缀的情况
        if '.SS' in code.upper() or '.SZ' in code.upper():
            return code.upper()
            
        # 美股判断：纯字母代码（如 AAPL, TSLA）或包含点号但非后缀（如 BRK.B）
        if code.replace('.', '').isalpha():
            return code.upper()
        
        # 去除可能的后缀
        code = code.replace('.SH', '').replace('.sh', '')
        
        # 根据代码前缀判断市场
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300', '430', '830', '870')): # 增加了北交所前缀以防万一
            return f"{code}.SZ"
        else:
            # 数字开头但无法识别的，默认当作深市处理（原有逻辑），或者可以考虑抛错
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市格式")
            return f"{code}.SZ"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据
        
        使用 yfinance.download() 获取历史数据
        
        流程：
        1. 转换股票代码格式
        2. 调用 yfinance API
        3. 处理返回数据
        """
        import yfinance as yf
        
        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)
        
        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")
        
        try:
            # 使用 yfinance 下载数据
            # multi_level_index=False 确保返回单层列索引（部分新版支持）
            try:
                df = yf.download(
                    tickers=yf_code,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
            except Exception as download_error:
                # 某些旧版不支持 auto_adjust
                logger.warning(f"yf.download 常规调用失败: {download_error}，尝试备选参数")
                df = yf.download(
                    tickers=yf_code,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

            if df is None or df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} ({yf_code}) 的数据")
            
            # 处理可能的 MultiIndex 列 (如果 download 返回了 (Price, Ticker) 格式)
            if isinstance(df.columns, pd.MultiIndex):
                # 如果是多层索引，尝试只取第一层级或者特定 Ticker
                # 通常单股下载只有一层，或者是 (Date, Close)
                try:
                    df = df.xs(yf_code, level=1, axis=1)
                except:
                    # 如果 xs 失败，可能是 structure 不同，直接 droplevel
                    df.columns = df.columns.get_level_values(0)

            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                # 尝试将 Date 列设为索引如果它存在
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                else:
                    # 如果这也不是，那数据结构可能有问题
                    pass

            return df
            
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            # 捕获所有异常并转换为 DataFetchError，包含原始错误信息
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {str(e)}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据
        
        yfinance 返回的列名：
        Open, High, Low, Close, Volume（索引是日期）
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # 重置索引，将日期从索引变为列
        df = df.reset_index()
        
        # 列名映射（yfinance 使用首字母大写）
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 计算涨跌幅（因为 yfinance 不直接提供）
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        
        # 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 平均价格
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        """
        获取实时行情 (Yahoo Finance)
        """
        import yfinance as yf
        yf_code = self._convert_stock_code(stock_code)
        
        try:
            ticker = yf.Ticker(yf_code)
            # fast_info usually provides faster access to current price
            info = ticker.fast_info
            # Some fields might be in ticker.info (more detailed but slower)
            
            price = info.last_price
            prev_close = info.previous_close
            
            if price is None or prev_close is None:
                 # Fallback to history if fast_info fails or market closed
                 hist = ticker.history(period="1d")
                 if not hist.empty:
                     price = hist['Close'].iloc[-1]
                     # If we only have 1d, we can't get prev_close easily unless we fetch more
                     # But fast_info should work for most US stocks
            
            change_amount = 0.0
            change_pct = 0.0
            
            if price and prev_close:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100
            
            # yfinance doesn't provide real-time turnover rate easily without float shares
            # We can try to get market cap
            mkt_cap = info.market_cap
            
            # Volume 
            # Note: fast_info doesn't always have current volume.
            # We might need detail info for that.
            
            # Construct Quote
            return RealtimeQuote(
                code=stock_code,
                name=stock_code, # Yahoo usually doesn't give short Chinese name
                price=round(price, 2) if price else 0.0,
                change_pct=round(change_pct, 2),
                change_amount=round(change_amount, 2),
                total_mv=mkt_cap if mkt_cap else 0.0,
                # Other fields blank for now
            )
            
        except Exception as e:
            logger.warning(f"Yahoo Finance realtime quote failed for {stock_code}: {e}")
            return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = YfinanceFetcher()
    
    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"获取失败: {e}")
