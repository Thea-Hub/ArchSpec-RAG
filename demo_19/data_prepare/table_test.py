import re
import json
from bs4 import BeautifulSoup
from typing import List, Any, Tuple


# ---------- 展开合并单元格 ----------
# 这是一个解析带合并单元格（colspan/rowspan）的 HTML 表格的函数，输入一个 <table> 标签，输出：
# matrix：把表格展开成整齐二维数组
# row_spans：每个单元格向下合并几行
# col_spans：每个单元格向右合并几列
def parse_table_with_colspan(table_tag):
    rows = table_tag.find_all('tr')
    if not rows:
        return [], [], []

    col_counts = []
    for row in rows:
        cols = row.find_all(['th', 'td'])
        col_count = 0
        for cell in cols:
            colspan = int(cell.get('colspan', 1))
            col_count += colspan
        col_counts.append(col_count)
    max_cols = max(col_counts) if col_counts else 0

    matrix = [[None] * max_cols for _ in range(len(rows))]
    row_spans = [[1] * max_cols for _ in range(len(rows))]
    col_spans = [[1] * max_cols for _ in range(len(rows))]

    for i, row in enumerate(rows):
        cols = row.find_all(['th', 'td'])
        col_index = 0
        for cell in cols:
            while col_index < max_cols and matrix[i][col_index] is not None:
                col_index += 1
            if col_index >= max_cols:
                break
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(strip=True)
            for r in range(i, min(i + rowspan, len(rows))):
                for c in range(col_index, min(col_index + colspan, max_cols)):
                    if matrix[r][c] is None:
                        matrix[r][c] = text
                        row_spans[r][c] = rowspan
                        col_spans[r][c] = colspan
            col_index += colspan
    return matrix, row_spans, col_spans


# ---------- 提取行维度表头名称（关键修改：去重+不重复拼接） ----------
def extract_row_header_names(matrix, header_rows, row_header_cols):
    row_header_names = []
    if not matrix or header_rows == 0 or row_header_cols == 0:
        return row_header_names

    # 行左侧分类列，只取【最顶层唯一表头】，避免重复前缀
    top_header = ""
    for row in range(header_rows):
        for col in range(row_header_cols):
            val = matrix[row][col]
            if val and val.strip():
                top_header = val.strip()
                break
        if top_header:
            break

    # 所有左侧列，统一用同一个顶层表头
    for _ in range(row_header_cols):
        row_header_names.append(top_header)
    return row_header_names


# ---------- 内容特征检测（备用） ----------
def auto_detect_table_structure_v2(matrix: List[List[Any]]) -> Tuple[int, int]:
    if not matrix or len(matrix) == 0:
        return 0, 0
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    if num_cols == 0:
        return 0, 0

    row_scores = []
    for i, row in enumerate(matrix):
        numeric_count = 0
        header_count = 0
        non_empty = 0
        for cell in row:
            if cell == '':
                continue
            non_empty += 1
            if is_numeric_cell(cell):
                numeric_count += 1
            if is_text_header_candidate(cell):
                header_count += 1
        if non_empty == 0:
            row_scores.append((i, 0))
            continue
        numeric_ratio = numeric_count / non_empty
        header_ratio = header_count / non_empty
        data_score = numeric_ratio - 0.5 * header_ratio
        row_scores.append((i, data_score))

    data_start = None
    for i, score in row_scores:
        if score > 0.3:
            data_start = i
            break
    if data_start is None:
        header_rows = 1
    else:
        header_rows = data_start

    max_header = max(1, num_rows // 2)
    if header_rows > max_header:
        header_rows = max_header
    if header_rows == 0:
        header_rows = 1

    if header_rows >= num_rows:
        row_header_cols = 0
    else:
        sample_end = min(header_rows + 10, num_rows)
        sample_rows = matrix[header_rows:sample_end]
        col_scores = []
        for col in range(num_cols):
            numeric_count = 0
            header_count = 0
            non_empty = 0
            for row in sample_rows:
                if col < len(row) and row[col] != '':
                    non_empty += 1
                    if is_numeric_cell(row[col]):
                        numeric_count += 1
                    if is_text_header_candidate(row[col]):
                        header_count += 1
            if non_empty == 0:
                col_scores.append(0)
                continue
            numeric_ratio = numeric_count / non_empty
            header_ratio = header_count / non_empty
            col_score = numeric_ratio - 0.5 * header_ratio
            col_scores.append(col_score)

        data_col_start = None
        for idx, score in enumerate(col_scores):
            if score > 0.3:
                data_col_start = idx
                break
        if data_col_start is None:
            row_header_cols = 1
        else:
            row_header_cols = data_col_start

        max_row_header = max(1, num_cols // 2)
        if row_header_cols > max_row_header:
            row_header_cols = max_row_header
        if row_header_cols == 0:
            row_header_cols = 1

    return header_rows, row_header_cols


# ---------- 辅助函数 ----------
def is_numeric_cell(cell: str) -> bool:
    if not isinstance(cell, str):
        cell = str(cell)
    cell = cell.strip()
    if not cell:
        return False
    cleaned = re.sub(r'[^0-9.-]', '', cell)
    if cleaned == '':
        return False
    try:
        float(cleaned)
        return True
    except:
        return False


def is_text_header_candidate(cell: str) -> bool:
    if not isinstance(cell, str):
        cell = str(cell)
    cell = cell.strip()
    if not cell:
        return False
    header_keywords = ['项目', '名称', '类别', '等级', '面积', '高度', '层数', '间距', '数量',
                       '指标', '参数', '类型', '耐火等级', '防火分区', '占地面积',
                       '序号', '编号', '指标名称', '单位', '分类', '规格', '火灾危险性', '最多允许层数']
    if any(kw in cell for kw in header_keywords):
        return True
    if re.match(r'^\d+\.', cell):
        return True
    if re.fullmatch(r'[\u4e00-\u9fa5]{1,10}', cell):
        return True
    return False


# ---------- 综合检测（优先使用合并单元格） ----------
def auto_detect_table_structure_with_merge(table_tag):
    matrix, row_spans, col_spans = parse_table_with_colspan(table_tag)
    if not matrix:
        return 0, 0, matrix, row_spans, col_spans

    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0

    max_rowspan_first = 1
    if num_rows > 0:
        for c in range(num_cols):
            if row_spans[0][c] > max_rowspan_first:
                max_rowspan_first = row_spans[0][c]
    header_rows = max_rowspan_first
    if header_rows > num_rows:
        header_rows = num_rows

    rowspan_cols = set()
    for row in range(min(header_rows, num_rows)):
        for col in range(num_cols):
            if row_spans[row][col] > 1:
                rowspan_cols.add(col)

    if rowspan_cols:
        max_rowspan_col = max(rowspan_cols)
        row_header_cols = max_rowspan_col + 1
        if row_header_cols > num_cols:
            row_header_cols = num_cols
    else:
        h, r = auto_detect_table_structure_v2(matrix)
        header_rows = h
        row_header_cols = r

    return header_rows, row_header_cols, matrix, row_spans, col_spans


# ---------- 构建多级列表头（核心修复：去重+层级组合） ----------
def build_multi_level_col_headers(matrix, header_rows, row_header_cols):
    if not matrix or header_rows == 0:
        return []
    num_cols = len(matrix[0])
    col_headers = []

    for col in range(row_header_cols, num_cols):
        col_parts = []
        for row in range(header_rows):
            val = matrix[row][col]
            if val and val.strip():
                col_parts.append(val.strip())

        # 超级关键修复：去重，不重复拼接相同文字
        unique_parts = []
        for p in col_parts:
            if p not in unique_parts:
                unique_parts.append(p)

        if unique_parts:
            col_header = ' '.join(unique_parts)
            col_headers.append(col_header)
        else:
            col_headers.append(f"列{col + 1}")
    return col_headers


# ---------- 构建语义化行标签（已优化） ----------
def build_dynamic_semantic_row_label(row_cells, row_header_names):
    if not row_header_names or not row_cells:
        return "未命名行"

    # 取出唯一顶层前缀
    base_name = row_header_names[0] if row_header_names else ""
    content_parts = []

    # 收集左侧所有层级单元格内容，过滤空/无效
    for cell in row_cells:
        cell_val = str(cell).strip() if cell else ""
        if cell_val and cell_val != "-":
            content_parts.append(cell_val)

    # 层级拼接：前缀 + 多级内容，无重复前缀
    if content_parts and base_name:
        return f"{base_name}：{'，'.join(content_parts)}"
    return "未命名行"


# ---------- 转换为嵌套 JSON（无改动） ----------
def convert_table_to_json(table_tag):
    header_rows, row_header_cols, matrix, _, _ = auto_detect_table_structure_with_merge(table_tag)
    if not matrix:
        return "[]"

    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    if row_header_cols >= num_cols:
        return "[]"

    row_header_names = extract_row_header_names(matrix, header_rows, row_header_cols)
    col_headers = build_multi_level_col_headers(matrix, header_rows, row_header_cols)

    data_rows = matrix[header_rows:]
    results = []

    for row in data_rows:
        row_label_cells = row[:min(row_header_cols, len(row))]
        row_label = build_dynamic_semantic_row_label(row_label_cells, row_header_names)

        data_dict = {}
        col_idx = 0
        for c in range(row_header_cols, min(num_cols, len(row))):
            if col_idx < len(col_headers):
                col_header = col_headers[col_idx]
                cell_value = str(row[c]).strip() if (c < len(row) and row[c]) else "-"
                data_dict[col_header] = cell_value
                col_idx += 1

        if data_dict:
            results.append({row_label: data_dict})

    return json.dumps(results, ensure_ascii=False, indent=2)


# ---------- 测试 ----------
if __name__ == "__main__":
    table_html = """<table><tr><td rowspan="2" colspan="2">建筑层数</td><td colspan="3">建筑的耐火等级</td></tr><tr><td>一、二级</td><td>三级</td><td>四级</td></tr><tr><td rowspan="3">地上楼层</td><td>1~2层</td><td>0.65</td><td>0.75</td><td>1.00</td></tr><tr><td>3层</td><td>0.75</td><td>1.00</td><td>-</td></tr><tr><td>≥4层</td><td>1.00</td><td>1.25</td><td>-</td></tr><tr><td rowspan="2">地下楼层</td><td>与地面出入口地面的高差 ΔH≤10m</td><td>0.75</td><td>-</td><td>-</td></tr><tr><td>与地面出入口地面的高差 ΔH&gt;10m</td><td>1.00</td><td>-</td><td>-</td></tr></table>
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if table:
        result_json = convert_table_to_json(table)
        print("解析结果：")
        print(result_json)

# 测试文本
# 3.2.1
# <table><tr><td rowspan="2" colspan="2">构件名称</td><td colspan="4">耐火等级</td></tr><tr><td>一级</td><td>二级</td><td>三级</td><td>四级</td></tr><tr><td rowspan="5">墙</td><td>防火墙</td><td>不燃性3.00</td><td>不燃性3.00</td><td>不燃性3.00</td><td>不燃性3.00</td></tr><tr><td>承重墙</td><td>不燃性3.00</td><td>不燃性2.50</td><td>不燃性2.00</td><td>难燃性0.50</td></tr><tr><td>楼梯间和前室的墙电梯井的墙</td><td>不燃性2.00</td><td>不燃性2.00</td><td>不燃性1.50</td><td>难燃性0.50</td></tr><tr><td>疏散走道两侧的隔墙</td><td>不燃性1.00</td><td>不燃性1.00</td><td>不燃性0.50</td><td>难燃性0.25</td></tr><tr><td>非承重外墙房间隔墙</td><td>不燃性0.75</td><td>不燃性0.50</td><td>难燃性0.50</td><td>难燃性0.25</td></tr><tr><td colspan="2">柱</td><td>不燃性3.00</td><td>不燃性2.50</td><td>不燃性2.00</td><td>难燃性0.50</td></tr><tr><td colspan="2">梁</td><td>不燃性2.00</td><td>不燃性1.50</td><td>不燃性1.00</td><td>难燃性0.50</td></tr><tr><td colspan="2">楼板</td><td>不燃性1.50</td><td>不燃性1.00</td><td>不燃性0.75</td><td>难燃性0.50</td></tr><tr><td colspan="2">屋顶承重构件</td><td>不燃性1.50</td><td>不燃性1.00</td><td>难燃性0.50</td><td>可燃性</td></tr></table>
# 3.3.2
# <table style="min-width: 1106px;"><colgroup><col style="min-width: 100px;"><col style="width: 106px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"></colgroup><tbody><tr><td colspan="2" rowspan="3" colwidth="0,106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>储存物品的火灾危险性类别</p></td><td colspan="1" rowspan="3" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>仓库的耐火等级</p></td><td colspan="1" rowspan="3" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>最多允许层数</p></td><td colspan="7" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>每座仓库的最大允许占地面积和每个防火分区的最大允许建筑面积(m2)</p></td></tr><tr><td colspan="2" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>单层仓库</p></td><td colspan="2" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>多层仓库</p></td><td colspan="2" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>高层仓库</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>地下或半地下仓库(包括地下或半地下室)</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>每座仓库</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>防火分区</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>每座仓库</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>防火分区</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>每座仓库</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>防火分区</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>防火分区</p></td></tr><tr><td colspan="1" rowspan="2" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>甲</p></td><td colspan="1" rowspan="1" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3、4项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>180</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>60</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1、2、5、6项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>750</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>250</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="4" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>乙</p></td><td colspan="1" rowspan="2" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1、3、4项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>900</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>300</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>250</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="2" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2、5、6项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>5</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2800</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>700</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>900</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>300</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="4" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>丙</p></td><td colspan="1" rowspan="2" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>5</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2800</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>700</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>150</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1200</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>400</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="2" colwidth="106" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2项</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>6000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4800</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1200</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>300</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2100</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>700</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1200</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>400</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr></tbody></table>
# 3.3.1
# <table style="min-width: 758px;"><colgroup><col style="min-width: 100px;"><col style="width: 114px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="min-width: 100px;"><col style="width: 144px;"></colgroup><tbody><tr><td colspan="1" rowspan="2" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>生产的火灾危险性类别</p></td><td colspan="1" rowspan="2" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>厂房的耐火等级</p></td><td colspan="1" rowspan="2" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>最多允许层数</p></td><td colspan="4" rowspan="1" colwidth="0,0,0,144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>每个防火分区的最大允许建筑面积(㎡)</p></td></tr><tr><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>单层厂房</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>多层厂房</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>高层厂房</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>地下或半地下厂房(包括地下或半地下室)</p></td></tr><tr><td colspan="1" rowspan="3" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>丙</p></td><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>6000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3000</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>500</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>8000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2000</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>500</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="3" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>丁</p></td><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4000</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1000</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>4000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>2000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>四级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="3" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>戊</p></td><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>一、二级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>不限</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>6000</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1000</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>三级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>5000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>3000</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr><tr><td colspan="1" rowspan="1" colwidth="114" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>四级</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>1500</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td><td colspan="1" rowspan="1" colwidth="144" data-border-top="true" data-border-right="true" data-border-bottom="true" data-border-left="true" text-align="left" style="text-align: left;"><p>-</p></td></tr></tbody></table>
#3.1.1 全文字
# <table><tr><td>生产的火灾危险性类别</td><td>使用或产生下列物质生产的火灾危险性特征</td></tr><tr><td>乙</td><td>1.闪点不小于28℃,但小于60℃的液体;2.爆炸下限不小于10%的气体;3.不属于甲类的氧化剂;4.不属于甲类的易燃固体;5.助燃气体;6.能与空气形成爆炸性混合物的浮游状态的粉尘、纤维、闪点不小于60℃的液体雾滴</td></tr><tr><td>丙</td><td>1.闪点不小于60℃的液体;2.可燃固体</td></tr><tr><td>丁</td><td>1.对不燃烧物质进行加工,并在高温或熔化状态下经常产生强辐射热、火花或火焰的生产;2.利用气体、液体、固体作为燃料或将气体、液体进行燃烧作其他用的各种生产;3.常温下使用或加工难燃烧物质的生产</td></tr><tr><td>戊</td><td>常温下使用或加工不燃烧物质的生产</td></tr></table>
# 5.5.17
# <table><tr><td rowspan="2" colspan="3">名称</td><td colspan="3">位于两个安全出口之间的疏散门</td><td colspan="3">位于袋形走道两侧或尽端的疏散门</td></tr><tr><td>一、二级</td><td>三级</td><td>四级</td><td>一、二级</td><td>三级</td><td>四级</td></tr><tr><td colspan="3">托儿所、幼儿园老年人建筑</td><td>25</td><td>20</td><td>15</td><td>20</td><td>15</td><td>10</td></tr><tr><td colspan="3">歌舞娱乐放映游艺场所</td><td>25</td><td>20</td><td>15</td><td>9</td><td>-</td><td>-</td></tr><tr><td rowspan="3">医疗建筑</td><td colspan="2">单、多层</td><td>35</td><td>30</td><td>25</td><td>20</td><td>15</td><td>10</td></tr><tr><td rowspan="2">高层</td><td>病房部分</td><td>24</td><td>-</td><td>-</td><td>12</td><td>-</td><td>-</td></tr><tr><td>其他部分</td><td>30</td><td>-</td><td>-</td><td>15</td><td>-</td><td>-</td></tr><tr><td rowspan="2">教学建筑</td><td colspan="2">单、多层</td><td>35</td><td>30</td><td>25</td><td>22</td><td>20</td><td>10</td></tr><tr><td colspan="2">高层</td><td>30</td><td>-</td><td>-</td><td>15</td><td>-</td><td>-</td></tr><tr><td colspan="3">高层旅馆、展览建筑</td><td>30</td><td>-</td><td>-</td><td>15</td><td>-</td><td>-</td></tr><tr><td rowspan="2">其他建筑</td><td colspan="2">单、多层</td><td>40</td><td>35</td><td>25</td><td>22</td><td>20</td><td>15</td></tr><tr><td colspan="2">高层</td><td>40</td><td>-</td><td>-</td><td>20</td><td>-</td><td>-</td></tr></table>
