def text_to_original_table(text: str) -> str:
    """
    把你检索到的 行标签+键值对 文本
    直接转换成【完全使用原始字段】的表格，不新增任何自定义表头
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return text

    # 1. 解析第一行：行标签 → 拆分成条件
    first_line = lines[0].replace("行标签：", "")
    conditions = {}
    for part in first_line.split("，"):
        if "：" in part:
            k, v = part.split("：", 1)
            conditions[k.strip()] = v.strip()

    # 2. 解析数据行
    values = {}
    for line in lines[1:]:
        if "：" in line:
            k, v = line.split("：", 1)
            # 自动提取后缀（如 单层厂房）
            short_key = k.split()[-1] if " " in k else k
            values[short_key.strip()] = v.strip()

    # 3. 组合成原始结构表格
    headers = list(conditions.keys()) + list(values.keys())
    row_data = list(conditions.values()) + list(values.values())

    # ====================== 在这里加一步：转美观表格 ======================
    return generate_pretty_table(headers, row_data)


# ====================== 【通用美观表格生成器】 ======================
def generate_pretty_table(headers, row_data):
    """自动计算宽度 + 生成整齐好看的纯文本表格"""
    # 计算每列最大宽度
    col_widths = []
    for h, d in zip(headers, row_data):
        width = max(len(str(h)), len(str(d))) + 2
        col_widths.append(width)

    # 绘制表格线
    def make_line():
        parts = ["-" * w for w in col_widths]
        return "+" + "+".join(parts) + "+"

    # 绘制一行内容
    def make_row(items):
        parts = []
        for item, w in zip(items, col_widths):
            parts.append(str(item).ljust(w - 1) + " ")
        return "|" + "|".join(parts) + "|"

    # 组合表格
    lines = []
    lines.append(make_line())
    lines.append(make_row(headers))
    lines.append(make_line())
    lines.append(make_row(row_data))
    lines.append(make_line())
    return "\n".join(lines)
# ================== 测试代码（用你这张表的HTML） ==================
if __name__ == "__main__":
    input_text = """
行标签：生产的火灾危险性类别：甲，厂房的耐火等级：一级，最多允许层数：宜采用单层
  每个防火分区的最大允许建筑面积(㎡) 单层厂房：4000
  每个防火分区的最大允许建筑面积(㎡) 多层厂房：3000
  每个防火分区的最大允许建筑面积(㎡) 高层厂房：- -
  每个防火分区的最大允许建筑面积(㎡) 地下或半地下厂房(包括地下或半地下室)：- -
"""
    print(text_to_original_table(input_text))