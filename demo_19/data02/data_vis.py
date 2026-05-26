import json
import re

def convert_unicode_to_chinese(input_file, output_file):
    """
    将JSON文件中的Unicode编码转换为中文字符
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 方法1: 使用json加载再保存（推荐，能处理所有Unicode转义）
        try:
            data = json.loads(content)
            
            # 保存为新的JSON文件，ensure_ascii=False确保中文正常显示
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, separators=(',', ': '))
            
            print(f"转换成功！文件已保存为: {output_file}")
            return
            
        except json.JSONDecodeError:
            # 如果JSON格式有问题，使用方法2
            print("JSON解析失败，尝试使用正则表达式方法...")
        
        # 方法2: 使用正则表达式替换Unicode编码（备用方法）
        def unicode_replacer(match):
            unicode_str = match.group(1)
            try:
                # 处理\u4e2d这种格式的Unicode
                return unicode_str.encode('utf-8').decode('unicode_escape')
            except:
                return unicode_str
        
        # 匹配所有\uXXXX格式的Unicode编码
        pattern = r'\\u([0-9a-fA-F]{4})'
        converted_content = re.sub(pattern, unicode_replacer, content)
        
        # 保存转换后的内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(converted_content)
        
        print(f"转换成功！文件已保存为: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")

def convert_specific_keys(data):
    """
    递归处理字典中的特定键值，将Unicode转换为中文
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                # 尝试解码Unicode转义序列
                try:
                    data[key] = value.encode('utf-8').decode('unicode_escape')
                except:
                    pass
            else:
                convert_specific_keys(value)
    elif isinstance(data, list):
        for item in data:
            convert_specific_keys(item)

# 使用方法
if __name__ == "__main__":
    input_filename = r"E:\PythonNotebook\juke ai\work project\l2\day19-RAG+微调实现智能专家系统（方案数据篇）\day19-RAG+微调实现智能专家系统（方案数据篇）\demo_19\storage\docstore.json"  # 替换为您的输入文件名
    output_filename = "converted_output_2.json"  # 输出文件名
    
    convert_unicode_to_chinese(input_filename, output_filename)
    
    # 验证转换结果
    try:
        with open(output_filename, 'r', encoding='utf-8') as f:
            converted_data = json.load(f)
        print("转换验证成功！转换后的部分内容预览：")
        
        # 预览部分转换后的内容
        preview_data = json.dumps(converted_data, ensure_ascii=False, indent=2)
        print(preview_data[:500] + "..." if len(preview_data) > 500 else preview_data)
        
    except Exception as e:
        print(f"验证失败: {e}")