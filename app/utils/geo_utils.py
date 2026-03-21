"""
上海市地址解析工具
将文本地址解析为所属行政区（3级降级策略）
"""

# 上海16个区
DISTRICT_LIST = [
    '黄浦区', '徐汇区', '长宁区', '静安区', '普陀区', '虹口区', '杨浦区',
    '宝山区', '闵行区', '嘉定区', '浦东新区', '松江区', '青浦区',
    '奉贤区', '金山区', '崇明区'
]

# 路名/地名 → 所属区（基于上海实际地理，可持续扩充）
ROAD_DISTRICT_MAP = {
    # 普陀区
    '华灵路': '普陀区', '光新路': '普陀区', '交通路': '普陀区',
    '真华路': '普陀区', '宜川路': '普陀区', '祁连山路': '普陀区',
    '志丹路': '普陀区', '岚皋路': '普陀区', '宜川': '普陀区',
    # 宝山区
    '沪太路': '宝山区', '场中路': '宝山区', '大场': '宝山区',
    '南大路': '宝山区', '华池路': '宝山区', '宝翔路': '宝山区',
    '建华路': '宝山区', '华江路': '宝山区', '祁村路': '宝山区',
    '联谊路': '宝山区', '泰和路': '宝山区',
    # 嘉定区
    '安亭': '嘉定区', '嘉松': '嘉定区', '嘉定': '嘉定区',
    '娄塘': '嘉定区', '南翔': '嘉定区', '陈翔路': '嘉定区',
    '昌吉路': '嘉定区', '曹安公路': '嘉定区',
    # 闵行区
    '七宝': '闵行区', '莘庄': '闵行区', '虹桥': '闵行区',
    # 静安区
    '静安': '静安区', '阳城路': '静安区', '灵石路': '静安区',
    # 浦东新区
    '浦东': '浦东新区', '周浦': '浦东新区', '外高桥': '浦东新区',
    '张江': '浦东新区', '川沙': '浦东新区',
    # 青浦区
    '青浦': '青浦区', '松泽': '青浦区', '徐泾': '青浦区',
    '朱家角': '青浦区',
    # 松江区
    '松江': '松江区', '佘山': '松江区',
    # 徐汇区
    '徐汇': '徐汇区', '漕河泾': '徐汇区', '田林': '徐汇区',
    # 长宁区
    '长宁': '长宁区', '虹古路': '长宁区',
    # 虹口区
    '虹口': '虹口区', '四川北路': '虹口区',
    # 杨浦区
    '杨浦': '杨浦区', '五角场': '杨浦区', '控江': '杨浦区',
    # 黄浦区
    '黄浦': '黄浦区', '南京路': '黄浦区', '外滩': '黄浦区',
    # 崇明区
    '崇明': '崇明区',
    # 金山区
    '金山': '金山区',
    # 奉贤区
    '奉贤': '奉贤区', '南桥': '奉贤区',
}

# 无效地址关键词（直接返回 None）
INVALID_KEYWORDS = {'无', '未知', '未备注', '自行', '家', '家中', '*', '0', '(跳过)'}


def extract_district(text: str) -> str | None:
    """
    从地址文本解析上海行政区
    返回：区名（如"宝山区"）或 None（无法识别）

    策略（优先级从高到低）：
    1. 直接包含区名（"宝山区"/"浦东新区"）
    2. 包含区名简写（"宝山"/"浦东"）
    3. 包含已知路名/地名，映射到区
    """
    if not text:
        return None

    text = str(text).strip()

    # 过滤无效值
    if text in INVALID_KEYWORDS or len(text) <= 1:
        return None

    # 1. 完整区名匹配
    for d in DISTRICT_LIST:
        if d in text:
            return d

    # 2. 路名/地名 → 区映射
    for keyword, district in ROAD_DISTRICT_MAP.items():
        if keyword in text:
            return district

    return None


def resolve_location(longitude, latitude, injury_location: str) -> dict:
    """
    综合解析地点信息，返回地点层级和区名

    Returns:
        {
            'level': 0/1/2,
            'district': '宝山区' or None,
            'has_coordinate': bool
        }
    """
    # Level 2: 已有经纬度
    if longitude and latitude:
        try:
            lng = float(longitude)
            lat = float(latitude)
            if 120 < lng < 122 and 30 < lat < 32:   # 上海经纬度范围
                # 由经纬度反查区（简化：用文本兜底）
                district = extract_district(injury_location)
                return {'level': 2, 'district': district, 'has_coordinate': True}
        except (ValueError, TypeError):
            pass

    # Level 1: 文本推断区
    district = extract_district(injury_location)
    if district:
        return {'level': 1, 'district': district, 'has_coordinate': False}

    # Level 0: 仅全市兜底
    return {'level': 0, 'district': None, 'has_coordinate': False}
