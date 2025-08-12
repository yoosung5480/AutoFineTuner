from tool.tool import greet_tool
from engine.engine import greet_engine

def main():
    print("=== Main Script Start ===")
    greet_tool()
    greet_engine()
    print("=== Main Script End ===")

if __name__ == "__main__":
    main()
