"""
pyinstaller --add-data .\icon.ico;. -F -w -i .\icon.ico pdf_merge.py
"""

import argparse
import os
import time
from pathlib import Path
from PyPDF2 import PdfFileMerger, PdfFileWriter, PdfFileReader
import tkinter
from tkinter import filedialog, messagebox


def main():
    parser = argparse.ArgumentParser(
        description="Simple pdf merge tool, Author:ZL",
        usage=
        f'python {os.path.basename(__file__)} -o <merged.pdf> <doc1.pdf> <doc2.pdf> ...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input_pdfs',
        nargs='*',
        metavar="FILE",
        help='input pdf files',
        type=str,
    )
    parser.add_argument(
        '-o'
        '--output',
        default='./merged.pdf',
        nargs='?',
        dest='output_pdf',
        metavar="FILE",
        help='name of output file',
        type=str,
    )
    args = parser.parse_args()
    out_path = Path(args.output_pdf).resolve()
    cwd = os.getcwd()
    pwd = Path(__file__).absolute().parent

    if not args.input_pdfs:
        gui = GUI(pwd)
        return

    args.input_pdfs = [
        str(Path(f).resolve()) for f in args.input_pdfs if f.endswith('.pdf')
    ]
    out_path = str(out_path)

    if len(args.input_pdfs) == 0:
        print("No pdf files")
        return

    MergePDF(args.input_pdfs, out_path)


class GUI():

    def __init__(self, pwd) -> None:
        window = tkinter.Tk()
        self.window = window
        window.title("pdf merge tool")
        window.geometry("350x200")
        window.iconbitmap(str(Path(pwd) / "icon.ico"))
        btn_open = tkinter.Button(window,
                                  text="打开文件",
                                  command=self.btn_open_callback)
        btn_open.pack(expand=tkinter.YES, fill=tkinter.BOTH, side=tkinter.TOP)

        self.btn_merge = tkinter.Button(self.window,
                                        text="合并",
                                        command=self.btn_merge_callback)
        self.btn_merge.pack(expand=tkinter.YES,
                            fill=tkinter.BOTH,
                            side=tkinter.TOP)
        self.btn_merge.config(state=tkinter.DISABLED)

        self.filelist = tkinter.Listbox(self.window)
        self.filelist.pack(expand=tkinter.YES,
                           fill=tkinter.BOTH,
                           side=tkinter.BOTTOM)

        self.window.mainloop()

    def btn_open_callback(self):
        self.fileName = filedialog.askopenfilenames(
            filetypes=[("PDF", ".PDF"), ("PDF", ".pdf")])

        self.filelist.delete(0, tkinter.END)
        for item in self.fileName:
            self.filelist.insert(0, item)

        if len(self.fileName) > 1:
            self.btn_merge.config(state=tkinter.NORMAL)

    def btn_merge_callback(self):
        print("push merge btn")
        self.btn_merge.config(state=tkinter.DISABLED)
        self.out_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                     filetypes=[("PDF", ".pdf")
                                                                ],
                                                     confirmoverwrite=True)
        if self.out_path:
            MergePDF(self.fileName, self.out_path)
            messagebox.showinfo("pdf merged",
                                f"您的PDF文件合并完成！\n输出文件为：{self.out_path}")
        self.btn_merge.config(state=tkinter.NORMAL)


# 使用os模块的walk函数，搜索出指定目录下的全部PDF文件
# 获取同一目录下的所有PDF文件的绝对路径
def getFileName(filedir):
    print(f"Searching dir{filedir}")
    file_list = [os.path.join(root, filespath) \
                 for root, dirs, files in os.walk(filedir) \
                 for filespath in files \
                 if str(filespath).endswith('pdf')
                 ]
    return file_list if file_list else []


# 合并同一目录下的所有PDF文件
def MergePDF(pdf_fileName: list, outfile: str):

    output = PdfFileWriter()
    outputPages = 0

    if pdf_fileName:
        for pdf_file in pdf_fileName:
            print(f"文件 {pdf_file} ", end='')
            # 读取源PDF文件
            input = PdfFileReader(open(pdf_file, "rb"), strict=False)

            # 获得源PDF文件中页面总数
            pageCount = input.getNumPages()
            outputPages += pageCount
            print(f"的页数为：{pageCount}")

            # 分别将page添加到输出output中
            for iPage in range(pageCount):
                output.addPage(input.getPage(iPage))

        print("合并后的总页数:%d." % outputPages)
        # 写入到目标PDF文件
        output.write(outfile)
        print(f"您的PDF文件合并完成！输出为：{outfile}")

    else:
        print("没有可以合并的PDF文件！")


if __name__ == '__main__':
    time1 = time.time()
    main()
    time2 = time.time()
    print('总共耗时：%s s.' % (time2 - time1))
