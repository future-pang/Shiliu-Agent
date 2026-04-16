import os
import asyncio
import httpx
import urllib.parse
from configs.settings import settings
from mcp.server.fastmcp import FastMCP
import datetime
import asyncio

mcp = FastMCP("QWeather-Service")
QWEATHER_API_KEY = settings.QWEATHER_API_KEY
if not QWEATHER_API_KEY:
    raise ValueError("未找到 QWEATHER_API_KEY，请检查 .env 文件！")

BASE_HOST = settings.qweather_base_url
GEO_URL = f"{BASE_HOST}/geo/v2/city/lookup"         # 地理编码/城市查询接口 (GeoAPI)
WEATHER_URL = f"{BASE_HOST}/v7/weather/now"         # 实时天气接口
INDICES_URL = f"{BASE_HOST}/v7/indices/1d"          # 生活指数/旅游建议接口
SUN_URL = f"{BASE_HOST}/v7/astronomy/sun"           # 日出日落
MOON_URL = f"{BASE_HOST}/v7/astronomy/moon"         # 月升月落及月相
FORECAST_URL = f"{BASE_HOST}/v7/weather/3d"         # 3d天气预报

async def _get_location_id(location_name: str) -> str | None:
    """
    内部辅助函数：将城市名转换为和风天气的 Location ID
    修复点：1. 中文URL编码 2. 完善异常分类 3. 更清晰的日志
    返回：有效ID / None（失败）
    """
    # 核心修复：对中文城市名做URL编码（解决400错误）
    encoded_location = urllib.parse.quote(location_name)
    params = {
        "location": encoded_location,
        "key": QWEATHER_API_KEY,
        "lang": "zh",   # 指定中文返回，避免乱码
        "number": 1     # 只返回第一个匹配结果（最精准）
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(GEO_URL, params=params)
            print(f"[GeoAPI 调试] 响应码: {resp.status_code}, 请求URL: {resp.url}")
            resp.raise_for_status()

            try:
                data = resp.json()
            except ValueError as e:
                print(f"[GeoAPI 错误] JSON解析失败: {e}, 响应内容: {resp.text}")
                return None

            if data.get("code") == "200" and data.get("location"):
                location_id = data["location"][0]["id"]
                print(f"[GeoAPI 成功] {location_name} → Location ID: {location_id}")
                return location_id
            else:
                err_code = data.get("code")
                err_msg = data.get("msg", "无错误信息")
                print(f"[GeoAPI 业务错误] 状态码: {err_code}, 信息: {err_msg}")
                return None

        except httpx.HTTPStatusError as e:
            print(f"[GeoAPI HTTP错误] 状态码: {e.response.status_code}, 内容: {e.response.text}")
            return None
        except httpx.TimeoutException:
            print(f"[GeoAPI 错误] 请求超时（{location_name}）")
            return None
        except Exception as e:
            print(f"[GeoAPI 未知错误] {str(e)}")
            return None


# 定义工具：获取实时天气
@mcp.tool()
async def get_current_weather(city: str) -> str:
    """
    查询指定城市的实时天气预报。
    参数 city: 城市名称（如：北京、峨眉山、乐山）
    """
    print(f"[Tool 调用] 正在查询 {city} 实时天气...")
    location_id = await _get_location_id(city)

    if not location_id:
        return f"抱歉，未能识别城市名「{city}」或 API 调用失败，请检查城市名称是否正确。"

    params = {"location": location_id, "key": QWEATHER_API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(WEATHER_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") == "200":
                now = data["now"]
                return (
                    f"{city} 当前实时天气：\n"
                    f"温度：{now['temp']}℃ (体感 {now['feelsLike']}℃)\n"
                    f"状况：{now['text']}\n"
                    f"风向：{now['windDir']}，风力 {now['windScale']} 级\n"
                    f"湿度：{now['humidity']}%\n"
                    f"更新时间：{data['updateTime']}"
                )
            else:
                return f"天气数据获取失败，业务状态码: {data.get('code')}，信息: {data.get('msg')}"

        except Exception as e:
            return f"天气接口调用异常：{str(e)}"


# 3. 定义工具：获取生活建议
@mcp.tool()
async def get_travel_advice(city: str) -> str:
    """
    获取针对该城市的生活指数建议，包括穿衣、紫外线、运动、旅游建议等。
    适合用于制定出行计划。
    """
    print(f"[Tool 调用] 正在获取 {city} 出行建议...")
    location_id = await _get_location_id(city)

    if not location_id:
        return f"无法获取「{city}」的生活建议，城市识别失败。"

    # type 1:运动, 2:洗车, 3:穿衣, 5:旅游
    params = {
        "location": location_id,
        "key": QWEATHER_API_KEY,
        "type": "1,3,5",
        "lang": "zh"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(INDICES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") == "200" and data.get("daily"):
                advice_list = [f"🔹 {i['name']}: {i['category']} - {i['text']}" for i in data["daily"]]
                return f"「{city}」出行建议：\n" + "\n".join(advice_list)
            elif data.get("code") == "200" and not data.get("daily"):
                return f"ℹ「{city}」暂无可用的生活指数数据。"
            else:
                return f"生活指数获取失败，状态码: {data.get('code')}，信息: {data.get('msg')}"

        except Exception as e:
            return f"生活指数接口调用异常：{str(e)}"


@mcp.tool()
async def get_astronomy_info(city: str, date: str = None) -> str:
    """
    查询指定城市的日出日落、月升月落及月相信息。
    参数 city: 城市名称
    参数 date: 查询日期，格式为 YYYYMMDD，默认为今天
    """

    if not date:
        date = datetime.datetime.now().strftime("%Y%m%d")

    location_id = await _get_location_id(city)
    if not location_id:
        return f"无法识别城市「{city}」，天文信息获取失败。"

    params = {
        "location": location_id,
        "key": QWEATHER_API_KEY,
        "date": date
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 1. 并发请求太阳和月亮数据
            sun_task = client.get(SUN_URL, params=params)
            moon_task = client.get(MOON_URL, params=params)
            sun_resp, moon_resp = await asyncio.gather(sun_task, moon_task)

            sun_data = sun_resp.json()
            moon_data = moon_resp.json()

            if sun_data.get("code") == "200" and moon_data.get("code") == "200":
                # 【核心修正】和风天气的天文数据在 daily 数组中，不是根节点
                sun_daily = sun_data.get("daily", [{}])[0]
                moon_daily = moon_data.get("daily", [{}])[0]

                # 提取正确的字段（对齐和风天气API返回结构）
                sunrise = sun_daily.get("sunrise", "未知")
                sunset = sun_daily.get("sunset", "未知")
                moonrise = moon_daily.get("moonrise", "未知")
                moonset = moon_daily.get("moonset", "未知")
                moon_phase = moon_daily.get("moonPhaseName", "未知")  # 月相中文名

                return (
                    f"{city} {date} 天文观测指南：\n"
                    f"【太阳】日出：{sunrise} | 日落：{sunset}\n"
                    f"【月亮】月升：{moonrise} | 月落：{moonset}\n"
                    f"【月相】当前阶段：{moon_phase}\n"
                    f"提示：该数据为当地时间，观测请注意天气变化。"
                )
            else:
                err_info = f"太阳接口状态码：{sun_data.get('code')}，月亮接口状态码：{moon_data.get('code')}"
                return f"天文数据获取失败，请检查服务权限。{err_info}"
        except Exception as e:
            return f"天文接口调用异常：{str(e)}"


# server/tools/weather_mcp.py

# ... (原有的常量保持不变)
WEATHER_URL = f"{BASE_HOST}/v7/weather/now"
FORECAST_URL = f"{BASE_HOST}/v7/weather/3d"  # 👈 新增：3天逐日预报


# ... (_get_location_id 等代码保持不变)

# 👇 新增工具：获取未来3天预报
@mcp.tool()
async def get_weather_forecast(city: str) -> str:
    """
    查询指定城市未来 3 天（含今天、明天、后天）的天气预报。
    """
    print(f"[Tool 调用] 正在查询 {city} 未来3天预报...")
    location_id = await _get_location_id(city)
    if not location_id:
        return f"抱歉，未能识别城市名「{city}」。"

    params = {"location": location_id, "key": QWEATHER_API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") == "200" and data.get("daily"):
                forecast_list = []
                for day in data["daily"]:
                    date = day.get("fxDate")
                    text_day = day.get("textDay")
                    temp_min = day.get("tempMin")
                    temp_max = day.get("tempMax")
                    forecast_list.append(f"📅 {date}: 白天{text_day}，气温 {temp_min}℃ ~ {temp_max}℃")

                return f"{city} 未来天气预报：\n" + "\n".join(forecast_list)
            else:
                return f"预报获取失败，状态码: {data.get('code')}"
        except Exception as e:
            return f"天气预报接口异常：{str(e)}"


if __name__ == "__main__":
    print("和风天气 MCP 服务启动中...")
    print(f"API Key: {QWEATHER_API_KEY[:6]}****{QWEATHER_API_KEY[-4:]}")
    print(f"专属 Host: {BASE_HOST}")
    print("=" * 50)
    mcp.run()