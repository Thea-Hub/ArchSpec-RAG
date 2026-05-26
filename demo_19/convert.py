# 扁平化数据
def convert_standard_to_flat(original_data):
    """
    将你提供的防火规范嵌套JSON 转换成 通用扁平结构化数据
    :param original_data: 你提供的原始规范列表
    :return: 标准结构化列表 [{}, {}, ...]
    """
    standard_list = []

    for item in original_data:
        # 遍历每一条规范（如 "建筑设计防火规范 3.4.1"）
        for full_title, content in item.items():
            # 拆分标题：规范名称 + 条文号
            if "建筑设计防火规范" in full_title:
                code_name = "建筑设计防火规范"
                clause = full_title.replace(code_name, "").strip()
            else:
                code_name = full_title
                clause = ""

            # 正文
            main_text = content.get("正文", "")
            table_content = content.get("表格", {}).get("内容", [])

            # 没有表格 → 只存文字
            if not table_content:
                standard_list.append({
                    "规范名称": code_name,
                    "条文号": clause,
                    "表格类型": "文字说明",
                    "正文": main_text,
                    "行维度": None,
                    "列维度": None,
                    "数值": None,
                    "单位": None
                })
                continue

            # 有表格 → 逐行解析
            for row_item in table_content:
                for row_key, col_dict in row_item.items():
                    # 解析行维度（自动识别：名称/层数/耐火等级/油量等）
                    row_dim = parse_row_dimension(row_key)

                    # 遍历列数据
                    for col_key, value in col_dict.items():
                        col_dim = parse_col_dimension(col_key)

                        # 数值清洗
                        try:
                            num_val = float(value)
                        except:
                            num_val = None

                        # 生成最终标准结构
                        standard_item = {
                            "规范名称": code_name,
                            "条文号": clause,
                            "表格类型": detect_table_type(clause, row_dim),
                            "正文": main_text,
                            "行维度": row_dim,
                            "列维度": col_dim,
                            "数值": num_val,
                            "单位": "m"  # 防火规范默认单位m，可自动识别扩展
                        }
                        standard_list.append(standard_item)

    return standard_list


# ------------------- 内部工具函数（自动解析行列）--------------------
def parse_row_dimension(row_key):
    """解析行维度：自动提取 建筑类型、层数、耐火等级、油量、储量等"""
    dim = {}
    key_lower = row_key.replace("：", ":").replace("名 称", "名称")

    if "名称" in key_lower:
        parts = [p.strip() for p in key_lower.split("，") if p.strip()]
        names = []
        for p in parts:
            if "名称" in p:
                val = p.split(":", 1)[1].strip()
                names.append(val)

        # 自动识别层数、耐火等级、变压器油量
        full_name = " ".join(names)
        dim["建筑类型"] = full_name

        if any(k in full_name for k in ["单、多层", "高层"]):
            dim["层数"] = next((k for k in ["单、多层", "高层"] if k in full_name), None)
        if any(k in full_name for k in ["一、二级", "三级", "四级"]):
            dim["耐火等级"] = next((k for k in ["一、二级", "三级", "四级"] if k in full_name), None)
        if "油量" in full_name:
            dim["变压器油量"] = full_name
        if "甲类储存物品" in full_name:
            dim["甲类分项"] = full_name

    return dim


def parse_col_dimension(col_key):
    """解析列维度：自动提取 对比对象类型、层数、耐火等级、道路类型等"""
    dim = {}
    parts = col_key.split(" ")
    dim["对象类型"] = parts[0] if parts else ""
    if len(parts) >= 2:
        dim["对象层数/等级"] = " ".join(parts[1:])
    return dim


def detect_table_type(clause, row_dim):
    """自动判断表格类型"""
    if "3.4" in clause or "3.5" in clause or "5.2" in clause:
        return "防火间距"
    if "3.2" in clause:
        return "耐火等级"
    if "3.3" in clause:
        return "防火分区面积"
    if "厂外铁路" in str(row_dim) or "道路" in str(row_dim):
        return "防火间距_铁路道路"
    return "通用表格"