import re
import json
from bs4 import BeautifulSoup
from typing import List, Any,Tuple


# ==================== 表格解析核心 ====================
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


def extract_row_header_names(matrix, header_rows, row_header_cols):
    row_header_names = []
    if not matrix or header_rows == 0 or row_header_cols == 0:
        return row_header_names

    # 左侧每一列单独拿表头（适配多列表头）
    for col in range(row_header_cols):
        col_header = ""
        for row in range(header_rows):
            cell_val = matrix[row][col]
            if cell_val and str(cell_val).strip():
                col_header = str(cell_val).strip()
                break
        row_header_names.append(col_header)
    return row_header_names


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
                       '序号', '编号', '指标名称', '单位', '分类', '规格', '火灾危险性', '最多允许层数','构件名称']
    if any(kw in cell for kw in header_keywords):
        return True
    if re.match(r'^\d+\.', cell):
        return True
    if re.fullmatch(r'[\u4e00-\u9fa5]{1,10}', cell):
        return True
    return False


def auto_detect_table_structure_v2(matrix: List[Any]) -> Tuple[int,int]:
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
    header_rows = 1 if data_start is None else data_start
    max_header = max(1, num_rows // 2)
    header_rows = min(header_rows, max_header)

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
                if col < len(row) and row[col] not in ("", None):
                    non_empty += 1
                    if is_numeric_cell(row[col]):
                        numeric_count += 1
                    if is_text_header_candidate(row[col]):
                        header_count += 1
            col_scores.append(numeric_count / non_empty - 0.5 * header_count / non_empty if non_empty else 0)
        data_col_start = next((idx for idx, s in enumerate(col_scores) if s > 0.3), None)
        row_header_cols = 1 if data_col_start is None else data_col_start
        row_header_cols = min(row_header_cols, max(1, num_cols // 2))
    return header_rows, row_header_cols


def auto_detect_table_structure_with_merge(table_tag):
    matrix, row_spans, col_spans = parse_table_with_colspan(table_tag)
    if not matrix:
        return 0, 0, matrix, row_spans, col_spans
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    max_rowspan_first = max(row_spans[0][c] for c in range(num_cols)) if num_rows > 0 else 1
    header_rows = min(max_rowspan_first, num_rows)
    rowspan_cols = {col for row in range(header_rows) for col in range(num_cols) if row_spans[row][col] > 1}
    if rowspan_cols:
        row_header_cols = max(rowspan_cols) + 1
    else:
        header_rows, row_header_cols = auto_detect_table_structure_v2(matrix)
    row_header_cols = min(row_header_cols, num_cols)
    return header_rows, row_header_cols, matrix, row_spans, col_spans


def build_multi_level_col_headers(matrix, header_rows, row_header_cols):
    if not matrix or header_rows == 0:
        return []
    num_cols = len(matrix[0])
    col_headers = []
    for col in range(row_header_cols, num_cols):
        parts = []
        for r in range(header_rows):
            val = matrix[r][col]
            if val and val.strip():
                parts.append(val.strip())
        unique = list(dict.fromkeys(parts))
        col_headers.append(" ".join(unique))
    return col_headers


def build_dynamic_semantic_row_label(row_cells, row_header_names):
    if not row_header_names or not row_cells:
        return "未命名行"

    # 用来存储最终的键值对：{ 表头名: 单元格值 }
    header_value_map = {}

    for idx, name in enumerate(row_header_names):
        if idx >= len(row_cells):
            continue

        cell_val = str(row_cells[idx]).strip() if row_cells[idx] else ""
        if not cell_val or cell_val == "-":
            continue

        # 关键：相同表头名 → 合并值（不重复显示表头）
        if name in header_value_map:
            # 如果值一样，不重复加
            if cell_val != header_value_map[name]:
                header_value_map[name] += f"，{cell_val}"
        else:
            header_value_map[name] = cell_val

    # 拼接成最终标签
    label_parts = [f"{k}：{v}" for k, v in header_value_map.items()]
    return "，".join(label_parts) if label_parts else "未命名行"


def convert_table_to_json(table_tag) -> List[dict]:
    header_rows, row_header_cols, matrix, _, _ = auto_detect_table_structure_with_merge(table_tag)
    if not matrix or row_header_cols >= len(matrix[0]):
        return []
    row_header_names = extract_row_header_names(matrix, header_rows, row_header_cols)
    # ========= 修复：补上 header_rows =========
    col_headers = build_multi_level_col_headers(matrix, header_rows, row_header_cols)
    data_rows = matrix[header_rows:]
    res = []
    for row in data_rows:
        label_cells = row[:row_header_cols]
        label = build_dynamic_semantic_row_label(label_cells, row_header_names)
        d = {}
        idx = 0
        for c in range(row_header_cols, len(row)):
            if idx < len(col_headers):
                d[col_headers[idx]] = str(row[c]).strip() if row[c] else "-"
                idx += 1
        if d:
            res.append({label: d})
    return res


def parse_table_new(html_str: str) -> List[dict]:
    soup = BeautifulSoup(html_str, 'html.parser')
    tbl = soup.find("table")
    return convert_table_to_json(tbl) if tbl else []


# ==================== 文本清洗 ====================
def clean_text(s):
    if not s:
        return ""
    s = re.sub(r'\s+', ' ', str(s)).strip()
    s = re.sub(r'\$(\d+(?:\.\d+)?)\\mathrm\{m\}\^2\$', r'\1㎡', s)
    s = re.sub(r'\$\\mathrm\{m\}\^2\$', '㎡', s)
    s = re.sub(r'\$(\d+(?:\.\d+)?)\\mathrm\{m\}\$', r'\1m', s)
    s = re.sub(r'\$1\s*/\s*3\$', '1/3', s)
    s = re.sub(r'\$1\s*/\s*2\$', '1/2', s)
    s = re.sub(r'\$(\d+(?:\.\d+)?)\\mathrm\{h\}\$', r'\1h', s)
    s = re.sub(r'\$([\d.]+)\$', r'\1', s)
    return s


# ==================== 主解析【最终修复核心】 ====================
def parse_spec_md_to_json(md_path, json_path, spec_name="建筑设计防火规范"):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 全局正则
    # 条文块：行首x.x.x 开头，直到下一个行首x.x.x
    block_pat = re.compile(
        r'^(#{0,3}\s*)?(\d+\.\d+\.\d+)\s+([\s\S]*?)(?=\n(?:#{0,3}\s*)?\d+\.\d+\.\d+|\Z)',
        re.MULTILINE
    )
    table_html_pat = re.compile(r'<table[\s\S]*?</table>')
    table_title_pat = re.compile(r'表\s+\d+\.\d+\.\d+[\s\S]*?(?=\n|$)')
    sub_table_title_pat = re.compile(r'续表\s*\d+\.\d+\.\d+')
    note_pat = re.compile(r'注：[\s\S]+?(?=\n(?:表|续表|<table|\d+\.\d+\.\d+)|\Z)')

    result = []

    for block in block_pat.finditer(content):
        full_block = block.group(0)
        code = block.group(2).strip()
        raw_body = block.group(3)

        # 2. 提取：表标题 / 续表标题 / 注释 / 表格
        table_titles = table_title_pat.findall(raw_body)
        sub_titles = sub_table_title_pat.findall(raw_body)
        note_list = note_pat.findall(raw_body)
        table_blocks = table_html_pat.findall(raw_body)

        # 合并字段
        main_title = table_titles[0].strip() if table_titles else ""
        sub_title = sub_titles[0].strip() if sub_titles else ""
        note_text = " ".join([n.strip() for n in note_list]).strip()

        # 3. 从正文里 删除 表标题、续表、注释、table，保证正文纯净
        pure_text = raw_body
        pure_text = table_title_pat.sub("", pure_text)
        pure_text = sub_table_title_pat.sub("", pure_text)
        pure_text = note_pat.sub("", pure_text)
        pure_text = table_html_pat.sub("", pure_text)
        pure_text = clean_text(pure_text)

        # 4. 解析所有表格
        all_tables = []
        for tbl_html in table_blocks:
            all_tables.extend(parse_table_new(tbl_html))

        # 5. 组装json结构
        item = {f"{spec_name} {code}": {"正文": pure_text}}
        if all_tables or main_title or sub_title or note_text:
            item[f"{spec_name} {code}"]["表格"] = {
                "标题": main_title,
                "续表标题": sub_title,
                "内容": all_tables,
                "注释": note_text
            }
        result.append(item)

    # 去重
    seen = set()
    final = []
    for d in result:
        k = next(iter(d.keys()))
        if k not in seen:
            seen.add(k)
            final.append(d)

    # 写出
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)

    print(f"✅ 解析完成 | 总条文：{len(final)}")
    print(f"✅ 已修复：表标题/续表/注释分离、行标签重复、条文错位")


# 运行
if __name__ == "__main__":
    INPUT_MD = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\data03\MinerU_markdown_202602082043429.md"
    OUTPUT_JSON = "规范解析结果_最终完美版.json"
    parse_spec_md_to_json(INPUT_MD, OUTPUT_JSON)
