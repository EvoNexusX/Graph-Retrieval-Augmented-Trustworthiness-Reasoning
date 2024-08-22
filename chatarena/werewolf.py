import subprocess

from chatarena import ROOT_DIR
# from graphrags.graphrag import index

# index.run()

# from GRAPHRAG import ROOT_DIR
# def init(root_path=ROOT_DIR):
#     command = [
#         "python","-m","graphrag.index",
#         "--root",root_path,
#         "--init"
#     ]
#     result = subprocess.run(command, capture_output=True, text=True)
#     # # 打印输出结果
#     print("标准输出:", result.stdout)
#     print("标准错误:", result.stderr)
def update(conversations,root_path=ROOT_DIR):

    str_conver = '\n'.join([conversation['content'] for conversation in conversations])
    with open(f"{root_path}/input/temp.txt",'w') as f:
        f.write(str_conver)
    command = [
        'python', '-m', 'graphrag.index',
        '--root',f"{root_path}"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    # print(result.stderr)
#
def query(question:str):

    # 定义命令和参数import os
    # from pathlib import Path
    #
    # FILE = Path(__file__).resolve()
    #
    # ROOT_DIR = os.path.dirname(FILE)
    command = [
        'python', '-m', 'graphrag.query',
        '--root', ROOT_DIR,
        '--method', 'local',
        question  # 这里不需要引号，因为这是一个列表中的单个字符串项
    ]

    #  运行命令
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印输出结果
    print("标准输出:", result.stdout)
    print("标准错误:", result.stderr)
    return result
# # init()
update([{"content":"玩家三说自己是狼人"}])
# query( f"You need to infer the identity and confidence of each player (except yourself) based on historical dialog and self-reflection. For one player takes up one line, and the output format is as follows:[Player] is inferred to be my [teammate/enemy]. My level of trust in him is [confidence] and his level of threat to me is [1 - confidence].\nExample:Player 5 is inferred to be my enemy. My level of trust in him is 0.324 and his level of threat to me is 0.676.\n不输出其他任何东西")


query('玩家三说了什么')