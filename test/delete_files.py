import os
import re
import shutil

def delete_matching_files_and_folders_walk(root_dir, pattern):
    """
    删除指定文件夹下所有文件名或文件夹名匹配特定正则表达式的文件和文件夹。
    在删除前，会列出所有匹配的文件和文件夹，并要求用户确认。
    
    :param root_dir: 要遍历的根目录
    :param pattern: 正则表达式模式，用于匹配文件名或文件夹名
    """
    regex = re.compile(pattern)
    to_delete = []

    # 遍历目录收集所有匹配的文件和文件夹
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            if regex.search(name):
                file_path = os.path.join(root, name)
                to_delete.append(file_path)
        
        for name in dirs:
            if regex.search(name):
                dir_path = os.path.join(root, name)
                to_delete.append(dir_path)
    breakpoint()
    # 如果没有找到匹配项，通知用户并退出
    if not to_delete:
        print("No matching files or directories found.")
        return

    # 显示所有将要删除的项
    print("The following items will be deleted:")
    for item in to_delete:
        print(item)

    # 请求用户确认
    confirm = input("Do you want to proceed with deletion? (y/n): ")
    if confirm.lower() == 'y':
        # 如果用户确认，执行删除
        for item in to_delete:
            if os.path.isdir(item):
                print(f"Deleting directory: {item}")
                shutil.rmtree(item)
            elif os.path.isfile(item):
                print(f"Deleting file: {item}")
                os.remove(item)
        print("Deletion completed.")
    else:
        print("Deletion cancelled.")

def delete_matching_files_and_folders_sub(root_dir, pattern):
    """
    删除指定文件夹下次一级的所有文件名或文件夹名匹配特定正则表达式的文件和文件夹。
    在删除前，会列出所有匹配的文件和文件夹，并要求用户确认。
    
    :param root_dir: 要检查的根目录
    :param pattern: 正则表达式模式，用于匹配文件名或文件夹名
    """
    regex = re.compile(pattern)
    to_delete = []

    # 只查看指定目录下的直接子文件和子文件夹
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if regex.search(entry.name):
                to_delete.append(entry.path)

    # 如果没有找到匹配项，通知用户并退出
    if not to_delete:
        print("No matching files or directories found.")
        return

    # 显示所有将要删除的项
    print("The following items will be deleted:")
    for item in to_delete:
        print(item)

    # 请求用户确认
    confirm = input("Do you want to proceed with deletion? (y/n): ")
    if confirm.lower() == 'y':
        # 如果用户确认，执行删除
        for item in to_delete:
            if os.path.isdir(item):
                print(f"Deleting directory: {item}")
                shutil.rmtree(item)
            elif os.path.isfile(item):
                print(f"Deleting file: {item}")
                os.remove(item)
        print("Deletion completed.")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    # 指定要遍历的根目录
    # root_directory = "/path/to/your/directory"
    # 指定要匹配的正则表达式
    # match_pattern = "A"  # 可以根据需要修改这个模式，例如'^A.*' 匹配所有以A开头的文件/文件夹
    
    root_directory = os.getenv("DIR", None)
    match_pattern = os.getenv("MATCH", None)
    
    if root_directory is None or match_pattern is None:
        print("Please specify the root directory and match pattern using the DIR and MATCH environment variables.")
    else:
        delete_matching_files_and_folders_sub(root_directory, match_pattern)
    
