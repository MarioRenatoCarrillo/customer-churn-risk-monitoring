import os

def print_tree(start_path, prefix=""):
    items = os.listdir(start_path)
    items.sort()

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        connector = "└── " if i == len(items) - 1 else "├── "

        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print("\nProject Structure:\n")
    print_tree(".")