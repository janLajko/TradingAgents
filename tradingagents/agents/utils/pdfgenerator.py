import markdown2
from pathlib import Path
import datetime

class PDFReportGenerator:
    def __init__(self, output_dir="reports_pdf", method="weasyprint"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        
        # 中文CSS样式
        self.css_style = """
        <style>
        @page {
            margin: 2cm;
            @bottom-center {
                content: "第 " counter(page) " 页";
                font-size: 10px;
                color: #666;
            }
        }
        
        body {
            font-family: "Microsoft YaHei", "PingFang SC", "SimHei", "Noto Sans SC", sans-serif;
            font-size: 12px;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }
        
        .header p {
            margin: 10px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }
        
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px;
            font-size: 20px;
            margin-top: 30px;
        }
        
        h2 { 
            color: #34495e; 
            border-left: 4px solid #3498db; 
            padding-left: 15px;
            margin-top: 25px;
            font-size: 16px;
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 0 5px 5px 0;
        }
        
        h3 { 
            color: #34495e;
            margin-top: 20px;
            font-size: 14px;
        }
        
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0;
            font-size: 11px;
        }
        
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left;
        }
        
        th { 
            background-color: #f1f3f4; 
            font-weight: bold;
            color: #2c3e50;
        }
        
        .highlight { 
            background-color: #fff3cd; 
            padding: 15px; 
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
        
        .decision-box {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #28a745;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .risk-warning {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 1px solid #dc3545;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .summary-box {
            background: linear-gradient(135deg, #cce5ff 0%, #b3d9ff 100%);
            border: 1px solid #007bff;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #007bff;
        }
        
        .page-break {
            page-break-before: always;
        }
        </style>
        """
    
    def generate_pdf(self, content, filename, title="交易分析报告"):
        """生成PDF报告，支持多种方法"""
        if not content:
            return None
        
        # 转换markdown为HTML
        html_content = markdown2.markdown(
            content, 
            extras=['tables', 'fenced-code-blocks', 'break-on-newline', 'strike']
        )
        
        # 添加页眉
        header_html = f"""
        <div class="header">
            <h1>{title}</h1>
            <p>生成时间: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        """
        
        # 组装完整HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            {self.css_style}
        </head>
        <body>
            {header_html}
            {html_content}
        </body>
        </html>
        """
        
        # 生成PDF
        output_path = self.output_dir / f"{filename}.pdf"
        
        try:
            if self.method == "weasyprint":
                return self._generate_with_weasyprint(full_html, output_path)
            elif self.method == "pdfkit":
                return self._generate_with_pdfkit(full_html, output_path)
            else:
                raise ValueError(f"不支持的方法: {self.method}")
        except Exception as e:
            print(f"PDF生成失败 ({self.method}): {e}")
            # 尝试备用方法
            try:
                backup_method = "pdfkit" if self.method == "weasyprint" else "weasyprint"
                print(f"尝试备用方法: {backup_method}")
                if backup_method == "weasyprint":
                    return self._generate_with_weasyprint(full_html, output_path)
                else:
                    return self._generate_with_pdfkit(full_html, output_path)
            except Exception as e2:
                print(f"备用方法也失败了: {e2}")
                return None
    
    def _generate_with_weasyprint(self, html_content, output_path):
        """使用WeasyPrint生成PDF"""
        try:
            import weasyprint
            
            # WeasyPrint配置
            html_doc = weasyprint.HTML(string=html_content, encoding='utf-8')
            html_doc.write_pdf(str(output_path))
            return output_path
        except ImportError:
            raise ImportError("请安装weasyprint: pip install weasyprint")
    
    def _generate_with_pdfkit(self, html_content, output_path):
        """使用pdfkit生成PDF"""
        try:
            import pdfkit
            
            options = {
                'page-size': 'A4',
                'encoding': 'UTF-8',
                'no-outline': None,
                'margin-top': '20mm',
                'margin-right': '20mm',
                'margin-bottom': '20mm',
                'margin-left': '20mm',
                'enable-local-file-access': None,
            }
            
            pdfkit.from_string(html_content, str(output_path), options=options)
            return output_path
        except ImportError:
            raise ImportError("请安装pdfkit: pip install pdfkit")
        except OSError:
            raise OSError("请安装wkhtmltopdf系统依赖")


# 使用示例
def example_usage():
    """使用示例"""
    # 创建PDF生成器（优先使用weasyprint）
    pdf_gen = PDFReportGenerator(
        output_dir="output_reports",
        method="weasyprint"  # 或 "pdfkit"
    )
    
    # 示例内容
    sample_content = """
# 谷歌(GOOGL) 交易分析报告

## 执行摘要
本报告对谷歌股票进行了全面分析...

## 基本面分析
- **收入增长**: 15.2%
- **市盈率**: 23.4
- **市净率**: 4.1

## 技术分析
当前价格显示上升趋势...

## 风险评估
主要风险因素包括...

## 投资建议
**建议**: 买入
**目标价格**: $150
"""
    
    # 生成PDF
    pdf_path = pdf_gen.generate_pdf(
        sample_content,
        "GOOGL_2025_07_10_analysis",
        "谷歌股票分析报告"
    )
    
    if pdf_path:
        print(f"PDF已生成: {pdf_path}")
    else:
        print("PDF生成失败")


if __name__ == "__main__":
    example_usage()