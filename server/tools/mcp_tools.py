import re
import asyncio
import datetime
import requests
from configs.settings import settings
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# 天气工具导入
from server.tools.weather_mcp import get_current_weather as fetch_weather
from server.tools.weather_mcp import get_travel_advice as fetch_advice
from server.tools.weather_mcp import get_astronomy_info as fetch_astronomy
from server.tools.weather_mcp import get_weather_forecast as fetch_forecast

# 高德地图工具导入
from server.tools.map_mcp import get_walking_plan as fetch_walking_plan
from server.tools.map_mcp import get_distance as fetch_distance
from server.tools.map_mcp import search_around as fetch_around_search
from server.tools.map_mcp import get_static_map as fetch_static_map

import requests
from langchain_core.tools import tool

def generate_image_tool(prompt: str) -> str:
    """
    【文生图工具】当用户要求绘图、生成图片或描述画面时使用。
    参数 prompt: 详细的中文画面描述提示词（限300字）。
    """
    # 测试文生图
    if any(k in prompt for k in ["彝族", "文创", "产品", "设计", "冰箱贴"]):
        print(f"[Trick] 拦截到大模型的提示词 '{prompt}'，正在强行替换为我们的高级 Prompt...")
        prompt = (
            "一张极度精美的彝族文创冰箱贴特写。主体是一个缩小的彝族传统漆器酒壶（公足）造型，"
            "材质呈现出黑红色高光漆器的温润质感。表面手工绘制有精细的红色与黄色火镰纹、云纹和标志性的鹰灵图腾。"
            "冰箱贴边缘镶嵌有哑光银饰，模仿大凉山银饰的复古质感。背景是极简的暖灰色，柔和的影棚光从侧面打过，"
            "突出3D立体浮雕纹路和细腻的漆面反光。4k分辨率，中国风文创设计感，精致，超现实主义摄影。"
        )

    img_conf = settings.image_llm
    url = f"{img_conf['base_url']}/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {img_conf['api_key']}"
    }
    payload = {
        "model": img_conf['model_id'],
        "prompt": prompt,
        "size": "2K",
        "response_format": "url",
        "sequential_image_generation": "disabled"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        res_json = response.json()
        if response.status_code == 200:
            img_url = res_json['data'][0]['url']
            return (
                f"【特工执行汇报】：图片已在后台生成完毕！系统已将图片推送到用户的屏幕上。[IMAGE_URL: {img_url}]\n\n"
                f"【极其重要的强制指令】：用户现在正看着这张图。请你立刻开始口头讲解设计亮点。"
                f"记住，你是个解说员，不是发图机器。不要在你的文字里写任何 URL，不要出现 '![]()'，直接用纯汉字夸它有多好看就行了！"
            )
        return f"生图失败: {res_json.get('error', {}).get('message', '未知错误')}"
    except Exception as e:
        return f"系统繁忙: {str(e)}"

# ==== DIY 工具 ====
@tool
def get_current_time() -> str:
    """
    获取当前的系统真实日期和时间
    当用户询问“今天”、“明天”、“后天”或特定日期，但你不确定当前日期时，请务必先调用此工具。
    """
    now = datetime.datetime.now()
    time_str = now.strftime('%Y-%m-%d %A')
    print(f"[Tool 调用] 正在获取系统时间: {time_str}")
    return f"当前真实日期是：{time_str}。请以此为基准推算用户的相对时间。"


@tool
def web_search(query: str) -> str:
    """
    通过搜索引擎查询互联网实时信息、突发新闻、景区最新政策及通用百科知识。
    """
    tavily_key = settings.TAVILY_API_KEY
    if not tavily_key:
        return "联网搜索由于未配置 TAVILY_API_KEY 而处于离线状态。请仅依靠本地知识。"

    print(f"[Tavily 联网搜索] 正在全网检索: {query}...")

    try:
        search_tool = TavilySearchResults(k=3, tavily_api_key=tavily_key)
        results = search_tool.invoke({"query": query})

        # 将多个网页结果拼装成一段可供特工阅读的背景知识
        formatted_res = []
        for i, res in enumerate(results):
            formatted_res.append(f"来源[{i + 1}]: {res['url']}\n内容: {res['content']}")

        return "\n\n".join(formatted_res)
    except Exception as e:
        return f"联网搜索暂时不可用: {str(e)}"

# ===== 天气 ====
@tool
def weather_api(city: str) -> str:
    """
    查询指定城市的实时天气预报和气温。
    参数 city: 城市名称（如：北京、峨眉山、乐山）
    """
    print(f" [Tool 调用] 正在真实请求 {city} 的和风天气 API...")
    return asyncio.run(fetch_weather(city))

@tool
def travel_advice_api(city: str) -> str:
    """
    获取针对该城市的生活指数建议，包括穿衣、紫外线、运动、旅游建议等。
    适合用于制定出行计划。
    """
    print(f"[Tool 调用] 正在真实请求 {city} 的出行建议 API...")
    return asyncio.run(fetch_advice(city))

@tool
def astronomy_api(city: str, date: str = None) -> str:
    """查询特定地点的日出日落、月升月落及当前月相，适用于摄影、夜游规划。"""
    print(f"[Tool 调用] 正在真实请求 {city} {date if date else '今日'} 的天文信息 API...")
    return asyncio.run(fetch_astronomy(city, date))

@tool
def weather_forecast_api(city: str) -> str:
    """
    查询指定城市未来3天（含今天、明天、后天）的天气预报。
    参数 city: 城市名称
    """
    print(f"[Tool 调用] 正在真实请求 {city} 的和风天气 3天预报 API...")
    return asyncio.run(fetch_forecast(city))

# ===== 高德 ====

@tool
def walking_plan_api(origin: str, destination: str) -> str:
    """
    查询起点到终点的步行规划路线（含距离、耗时、详细步骤）。
    参数 origin: 起点地址（如：乐山大佛景区）
    参数 destination: 终点地址（如：峨眉山游客中心）
    """
    print(f"[Tool 调用] 正在请求 {origin} → {destination} 的步行规划 API...")
    return asyncio.run(fetch_walking_plan(origin, destination))

@tool
def distance_api(origin: str, destination: str, type: str = "straight") -> str:
    """
    测量两点间的距离（支持直线/驾车/步行）。
    参数 origin: 起点地址
    参数 destination: 终点地址
    参数 type: 距离类型（straight=直线, driving=驾车, walking=步行），默认直线
    """
    print(f"[Tool 调用] 正在测量 {origin} ↔ {destination}（{type}）距离 API...")
    return asyncio.run(fetch_distance(origin, destination, type))

@tool
def around_search_api(address: str, keyword: str, radius: int = 1000, page_size: int = 10) -> str:
    """
    搜索指定地址周边的POI（如餐厅、酒店、加油站等）。
    参数 address: 中心地址（如：成都市锦江区春熙路）
    参数 keyword: 搜索关键词（如：火锅、酒店、加油站）
    参数 radius: 搜索半径（米），默认1000米
    参数 page_size: 返回结果数量，默认10条
    """
    print(f"[Tool 调用] 正在搜索 {address} 周边{radius}米内「{keyword}」的 POI API...")
    return asyncio.run(fetch_around_search(address, keyword, radius, page_size))

@tool
def static_map_api(address: str, zoom: int = 15, width: int = 600, height: int = 400) -> str:
    """
    生成指定地址的静态地图图片URL（可直接在浏览器打开/嵌入页面）。
    参数 address: 地图中心地址
    参数 zoom: 地图缩放级别（1-19），默认15（街道级）
    参数 width: 图片宽度（像素），默认600
    参数 height: 图片高度（像素），默认400
    """
    print(f"[Tool 调用] 正在生成 {address} 的静态地图 URL API...")
    return asyncio.run(fetch_static_map(address, zoom, width, height))
