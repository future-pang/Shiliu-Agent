import httpx
import urllib.parse
from mcp.server.fastmcp import FastMCP
from configs.settings import settings

mcp = FastMCP("AMap-Service")
AMAP_API_KEY = settings.AMAP_API_KEY
if not AMAP_API_KEY:
    raise ValueError("未找到 AMAP_API_KEY，请检查 .env 文件！")

BASE_AMAP_URL = settings.amap_base_url
GEOCODE_URL = f"{BASE_AMAP_URL}/geocode/geo"                # 地理编码（地址→经纬度）
WALKING_PLAN_URL = f"{BASE_AMAP_URL}/direction/walking"     # 步行规划
DISTANCE_URL = f"{BASE_AMAP_URL}/distance"                  # 距离测量
AROUND_SEARCH_URL = f"{BASE_AMAP_URL}/place/around"         # 周边搜索
STATIC_MAP_URL = f"{BASE_AMAP_URL}/staticmap"               # 静态地图


async def _get_lng_lat(address: str) -> tuple[str, str] | None:
    """
    内部辅助函数：地址转经纬度（和天气工具的 _get_location_id 对应）
    返回：(经度, 纬度) / None（失败）
    """
    if "," in address and len(address.split(",")) == 2:
        try:
            lng, lat = address.split(",")
            float(lng)
            float(lat)
            print(f"[AMap 调试] 输入为经纬度，直接返回：{lng},{lat}")
            return lng, lat
        except ValueError:
            pass

    params = {
        "address": address,
        "key": AMAP_API_KEY,
        "output": "json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(GEOCODE_URL, params=params)
            print(f"[AMap 地理编码] 响应码: {resp.status_code}, 请求地址: {address}")
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1" and len(data.get("geocodes", [])) > 0:
                lng_lat = data["geocodes"][0]["location"].split(",")
                lng, lat = lng_lat[0], lng_lat[1]
                print(f"[AMap 地理编码成功] {address} → 经度:{lng}, 纬度:{lat}")
                return lng, lat
            else:
                err_info = data.get("info", "无错误信息")
                print(f"[AMap 地理编码失败] {address} → 错误信息: {err_info}")
                return None
        except Exception as e:
            print(f"[AMap 地理编码异常] {address} → {str(e)}")
            return None


# 步行规划（起点→终点的步行路线）
@mcp.tool()
async def get_walking_plan(origin: str, destination: str) -> str:
    """
    查询起点到终点的步行规划路线（含距离、耗时、详细步骤）
    参数 origin: 起点地址（如：北京市朝阳区天安门）
    参数 destination: 终点地址（如：北京市朝阳区故宫）
    """
    print(f"[AMap 工具] 正在规划 {origin} → {destination} 步行路线...")

    # 转换起点/终点为经纬度
    origin_lng_lat = await _get_lng_lat(origin)
    dest_lng_lat = await _get_lng_lat(destination)
    if not origin_lng_lat or not dest_lng_lat:
        return f"步行规划失败：\n- 起点「{origin}」或终点「{destination}」无法识别，请检查地址是否正确。"

    # 拼接经纬度参数（高德格式：lng,lat）
    origin_str = f"{origin_lng_lat[0]},{origin_lng_lat[1]}"
    dest_str = f"{dest_lng_lat[0]},{dest_lng_lat[1]}"

    params = {
        "origin": origin_str,
        "destination": dest_str,
        "key": AMAP_API_KEY,
        "output": "json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(WALKING_PLAN_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1" and data.get("route"):
                route = data["route"]["paths"][0]

                if data.get("status") == "1" and data.get("route"):
                    route = data["route"]["paths"][0]
                    steps = route["steps"]

                    # 如果步骤超过4步，直接省略中间详情，大幅节约 Token
                    if len(steps) > 4:
                        step_list = [f"{i + 1}. {step['instruction']}（{step['distance']}米）" for i, step in
                                     enumerate(steps[:2])]
                        step_list.append("... (中间路段较长，省略详细导航) ...")
                        step_list.append(f"{len(steps)}. {steps[-1]['instruction']}（{steps[-1]['distance']}米）")
                    else:
                        step_list = [f"{i + 1}. {step['instruction']}（{step['distance']}米）" for i, step in
                                     enumerate(steps)]

                return (
                        f"{origin} → {destination} 步行规划：\n"
                        f"总距离：{route['distance']} 米\n"
                        f"预计耗时：{route['duration']} 秒（约 {int(route['duration']) / 60:.1f} 分钟）\n"
                        f"步行步骤：\n" + "\n".join(step_list)
                )
            else:
                err_info = data.get("info", "无错误信息")
                return f"步行规划失败：{err_info}"
        except Exception as e:
            return f"步行规划接口异常：{str(e)}"


# 距离测量（两点间的直线/驾车/步行距离）
@mcp.tool()
async def get_distance(origin: str, destination: str, type: str = "straight") -> str:
    """
    测量两点间的距离（支持直线/驾车/步行）
    参数 origin: 起点地址
    参数 destination: 终点地址
    参数 type: 距离类型（straight=直线, driving=驾车, walking=步行），默认直线
    """
    print(f"[AMap 工具] 正在测量 {origin} ↔ {destination}（{type}）距离...")

    # 转换起点/终点为经纬度
    origin_lng_lat = await _get_lng_lat(origin)
    dest_lng_lat = await _get_lng_lat(destination)
    if not origin_lng_lat or not dest_lng_lat:
        return f"距离测量失败：\n- 起点「{origin}」或终点「{destination}」无法识别，请检查地址是否正确。"

    # 拼接参数（高德 distance 接口格式：origins=lng1,lat1&destinations=lng2,lat2）
    origins = f"{origin_lng_lat[0]},{origin_lng_lat[1]}"
    destinations = f"{dest_lng_lat[0]},{dest_lng_lat[1]}"

    # 映射距离类型到高德接口参数
    type_map = {
        "straight": "1",    # 直线距离
        "driving": "0",     # 驾车距离
        "walking": "2"      # 步行距离
    }
    amap_type = type_map.get(type, "1")

    params = {
        "origins": origins,
        "destination": destinations,
        "type": amap_type,
        "key": AMAP_API_KEY,
        "output": "json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(DISTANCE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1" and len(data.get("results", [])) > 0:
                result = data["results"][0]
                distance = int(result["distance"])
                unit = "米" if distance < 1000 else "千米"
                distance_show = distance if unit == "米" else f"{distance / 1000:.2f}"

                return (
                    f"{origin} ↔ {destination} 距离测量（{type}）：\n"
                    f"距离：{distance_show} {unit}\n"
                    f"⏱预计耗时（仅驾车/步行）：{result.get('duration', '无')} 秒"
                )
            else:
                err_info = data.get("info", "无错误信息")
                return f"距离测量失败：{err_info}"
        except Exception as e:
            return f"距离测量接口异常：{str(e)}"


# 周边搜索
@mcp.tool()
async def search_around(address: str, keyword: str, radius: int = 1000, page_size: int = 5) -> str:
    """
    搜索指定地址周边的POI（如餐厅、酒店、加油站等）
    参数 address: 中心地址（如：成都市锦江区春熙路）
    参数 keyword: 搜索关键词（如：火锅、酒店、加油站）
    参数 radius: 搜索半径（米），默认1000米
    参数 page_size: 返回结果数量，默认10条
    """
    print(f"[AMap 工具] 正在搜索 {address} 周边{radius}米内的「{keyword}」...")

    # 转换中心地址为经纬度
    center_lng_lat = await _get_lng_lat(address)
    if not center_lng_lat:
        return f"周边搜索失败：\n- 中心地址「{address}」无法识别。"

    # 拼接参数
    location = f"{center_lng_lat[0]},{center_lng_lat[1]}"

    params = {
        "location": location,
        "keywords": keyword,
        "radius": radius,
        "page_size": page_size,
        "output": "json",
        "key": AMAP_API_KEY
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(AROUND_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1" and len(data.get("pois", [])) > 0:
                pois = data["pois"][:page_size]

                poi_list = [
                    f"{i + 1}. {poi['name']}\n  📍 地址：{poi['address']}\n  📞 电话：{poi.get('tel', '无')}"
                    for i, poi in enumerate(pois)
                ]

                return (
                        f"{address} 周边{radius}米内的「{keyword}」（共{data['count']}条，显示前{page_size}条）：\n"
                        + "\n\n".join(poi_list)
                )
            else:
                err_info = data.get("info", "无错误信息")
                return f"周边搜索失败：{err_info}（未找到「{keyword}」相关结果）"
        except Exception as e:
            return f"周边搜索接口异常：{str(e)}"


# 静态地图（生成指定位置的静态地图图片URL）
@mcp.tool()
async def get_static_map(address: str, zoom: int = 15, width: int = 600, height: int = 400) -> str:
    """
    生成指定地址的静态地图图片URL（可直接在浏览器打开/嵌入页面）
    参数 address: 地图中心地址
    参数 zoom: 地图缩放级别（1-19），默认15（街道级）
    参数 width: 图片宽度（像素），默认600
    参数 height: 图片高度（像素），默认400
    """
    print(f"[AMap 工具] 正在生成 {address} 的静态地图...")

    # 转换中心地址为经纬度
    center_lng_lat = await _get_lng_lat(address)
    if not center_lng_lat:
        return f"静态地图生成失败：\n- 中心地址「{address}」无法识别，请检查地址是否正确。"

    # 拼接参数（高德静态地图接口参数）
    location = f"{center_lng_lat[0]},{center_lng_lat[1]}"
    params = {
        "location": location,
        "zoom": zoom,
        "size": f"{width}*{height}",
        "markers": f"mid,0xFF0000,A:{location}",  # 红色标记点
        "key": AMAP_API_KEY
    }

    # 生成完整URL
    try:
        query_string = urllib.parse.urlencode(params)
        map_url = f"{STATIC_MAP_URL}?{query_string}"
        return (
            f"{address} 静态地图生成成功：\n"
            f"地图URL：{map_url}\n"
            f"提示：复制URL到浏览器即可查看，缩放级别{zoom}，尺寸{width}x{height}像素。"
        )
    except Exception as e:
        return f"静态地图URL生成异常：{str(e)}"


if __name__ == "__main__":
    print("高德地图 MCP 服务启动中...")
    print(f"API Key: {AMAP_API_KEY[:6]}****{AMAP_API_KEY[-4:]}")
    print("=" * 50)
    mcp.run()





