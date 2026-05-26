#  =====================测试模块一:解析表格=======================
# 逻辑总结：
# 1.先算清表格真实行列；
# 2.建空矩阵；（此时是最大行数*最大列数）
# 3.逐单元格填文字 + 处理合并；（从第一行取所有单元格内容，先向右合并再向下合并，合并好了把这个单元格的文字，填到每一个被合并覆盖的位置上，再开始下一行）
# 4.返回整齐表格。
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

    for i, row in enumerate(rows): # rows = [行0, 行1, 行2]，i=0,1,2,enumerate(rows)会把它变成带编号的「元组」序列,实现自动计数 + 返回元素-->(0, 行0), (1, 行1), (2, 行2)
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

from bs4 import BeautifulSoup
# ====================测试=============================
# 1. 把 HTML 表格包在 三引号 里 → 变成字符串
html = """
<table>
<tr><td rowspan="2" colspan="3">类 别</td><td colspan="3">固定顶储罐</td>
<td rowspan="2">浮顶储罐或设置充氮保护设备的储罐</td>
<td rowspan="2">卧式储罐</td></tr>
<tr><td>地上式</td><td>半地下式</td><td>地下式</td></tr>
<tr><td rowspan="2">甲、乙类液体储罐</td>
<td rowspan="3">单罐容量V(m3)</td><td>V≤1000</td><td>0.75D</td>
<td rowspan="2">0.5D</td><td rowspan="2">0.4D</td><td rowspan="2">0.4D</td>
<td rowspan="3">≥0.8m</td></tr>
<tr><td>V&gt;1000</td><td>0.6D</td></tr>
<tr><td>丙类液体储罐</td><td>不限</td><td>0.4D</td><td>不限</td><td>不限</td><td>-</td>
</tr>
</table>
"""

# 2. 解析成 BeautifulSoup 对象
soup = BeautifulSoup(html, "html.parser")

# 3. 拿到 table 标签 → 这才是合法的 table_tag！
table_tag = soup.find("table")

# 4. 调用你的函数（把你之前的函数放这里就能跑）
matrix, row_spans, col_spans = parse_table_with_colspan(table_tag)

# 打印看结果（最终生成的表格）
for row in matrix:
    print(row)

print(matrix, row_spans, col_spans)