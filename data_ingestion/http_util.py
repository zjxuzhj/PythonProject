import requests
import json
import getAllStockCsv
import os

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

def get_json_by_get(url, params=None, headers=None, timeout=10, proxies=None, session=None, cookies=None):
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    try:
        sess = session or requests
        resp = sess.get(url, params=params, headers=merged_headers, timeout=timeout, proxies=proxies, allow_redirects=True, cookies=cookies)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return json.loads(resp.text)
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

def _parse_cookie_header(cookie_header: str):
    if not cookie_header:
        return None
    parts = [p.strip() for p in str(cookie_header).split(";") if p.strip()]
    out = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[k] = v.strip()
    return out or None

def get_ztlive_json(timeout=10, proxies=None, cookie=None):
    url = "https://duanxianxia.cn/vendor/livedata/ztlive.json"
    ua = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Mobile Safari/537.36"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Cache-Control": "max-age=0",
        "User-Agent": ua,
        "Referer": "https://duanxianxia.cn/",
        "Origin": "https://duanxianxia.cn",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": "\"Android\"",
    }
    try:
        session = requests.Session()
        cookie_header = cookie if cookie is not None else os.environ.get("DUANXIANXIA_COOKIE", "")
        cookies = _parse_cookie_header(cookie_header)
        if cookies:
            session.cookies.update(cookies)
            headers["Cookie"] = "; ".join([f"{k}={v}" for k, v in cookies.items()])
        session.get("https://duanxianxia.cn/", headers={**DEFAULT_HEADERS, **headers}, timeout=timeout, proxies=proxies, allow_redirects=True)
        return get_json_by_get(url, headers=headers, timeout=timeout, proxies=proxies, session=session, cookies=session.cookies)
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}

def updateZTThemeAndTime():
    # 1. 发送POST请求获取数据
    url = "https://duanxianxia.cn/api/getZtliveData"
    payload = {
        "type": "plate",  # 动态参数：复盘类型（如'plate'表示按板块）
        # "date": "2025-06-21",  # 动态参数：日期（格式为YYYY-MM-DD）
        "from": "web"  # 固定参数，标识请求来源
    }
    query_tool = getAllStockCsv.StockQuery()
    response = requests.post(url, data=payload)

    if response.status_code == 200:
        # 2. 解析JSON响应
        data = response.json()

        if data.get("result") == "success":
            # 4. 转换为Python对象
            stock_data = {
                "result": data["result"],
                "url": data["url"],
                "stocks": data["list"],
            }

            # 5. 使用数据（示例：打印前5个股票）
            print("解析成功！股票数量：", len(stock_data["stocks"]))
            print("前5个股票信息：")
            for stock in stock_data["stocks"]:
                # print(
                #     f"代码：{stock['code']} | 名称：{stock['name']} | 原因：{stock['ztyy']} | 时间：{stock['time']} | 状态：{stock['zt']}")
                query_tool.add_time(getAllStockCsv.code_add_prefix(stock['code']), stock['time'])
                query_tool.add_theme(getAllStockCsv.code_add_prefix(stock['code']), stock['ztyy'])
        else:
            print("接口返回失败：", data)
    else:
        print(f"请求失败，状态码：{response.status_code}")

def updateZTThemeAndTimeByGet(timeout=10, proxies=None, cookie=None):
    query_tool = getAllStockCsv.StockQuery()
    data = get_ztlive_json(timeout=timeout, proxies=proxies, cookie=cookie)

    if isinstance(data, dict) and data.get("result") == "success":
        stocks = data.get("list") or []
        print("解析成功！股票数量：", len(stocks))
        print("前5个股票信息：")
        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            code = stock.get("code")
            if not code:
                continue
            code_with_prefix = getAllStockCsv.code_add_prefix(str(code))
            t = stock.get("time", "")
            theme = stock.get("ztyy", "")
            if t != "":
                query_tool.add_time(code_with_prefix, t)
            if theme != "":
                query_tool.add_theme(code_with_prefix, theme)
    else:
        print("接口返回失败：", data)

if __name__ == "__main__":
    updateZTThemeAndTimeByGet()

