WITH ordered AS (
    SELECT date, close,
           LAG(close) OVER (ORDER BY date) AS prev_close
    FROM stock_prices
    WHERE ticker = 'GOOGL'
)
SELECT date, 100.0 * (close - prev_close) / prev_close AS daily_return_pct
FROM ordered
WHERE prev_close IS NOT NULL
ORDER BY daily_return_pct ASC, date ASC
LIMIT 1;
