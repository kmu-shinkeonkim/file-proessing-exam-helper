import random
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# Configuration Constants
# ============================================================================

# ===== Display Settings =====
SKIP_PROGRESS = True
#   - True:  각 테스트에서 최종 결과만 표시
#   - False: 모든 insert/delete 단계마다 트리 상태를 표시

SHOW_ALL_IN_ONE_TIME = False
#   - True:  모든 결과를 한번에 출력
#   - False: 각 섹션(테스트 케이스, BST, AVL, BT)마다 Enter 키를 눌러야 다음으로 진행
#            (SKIP_PROGRESS=False와 함께 사용시 각 insert/delete 단계마다도 대기)

# ===== Executable Paths =====
# 각 트리 타입별 실행 파일 경로를 지정하세요 (submit 디렉토리 기준 상대 경로)
# None으로 설정하면 해당 트리 테스트를 건너뜁니다
BST_EXECUTABLE = "BST"
AVL_EXECUTABLE = "AVL"
BTREE_EXECUTABLE = "BT"

# ===== Test Case Settings =====
# CUSTOM_INSERT_VALUES:
#   - None: 랜덤 생성
#   - List[int]: 직접 지정한 값 사용 (예: [10, 5, 15, 3, 7])
CUSTOM_INSERT_VALUES = None

# CUSTOM_DELETE_VALUES:
#   - None: CUSTOM_INSERT_VALUES에서 랜덤 선택
#   - List[int]: 직접 지정한 값 사용 (예: [5, 15])
CUSTOM_DELETE_VALUES = None

# 랜덤 생성 시 사용할 크기 (CUSTOM_INSERT_VALUES가 None일 때만 사용)
RANDOM_CREATE_COUNT = 15
RANDOM_DELETE_COUNT = 8


@dataclass
class TestCase:
  """Represents a test case with insert and delete operations"""
  create_count: int
  delete_count: int
  created_numbers: List[int]
  deleted_numbers: List[int]

  @classmethod
  def generate(
    cls,
    custom_insert: Optional[List[int]] = None,
    custom_delete: Optional[List[int]] = None,
    create_count: int = 15,
    delete_count: int = 8,
    seed: Optional[int] = None
  ):
    """
    Generate a test case

    Args:
      custom_insert: Custom insert values (if None, generate randomly)
      custom_delete: Custom delete values (if None, select from custom_insert randomly)
      create_count: Number of random inserts (used only if custom_insert is None)
      delete_count: Number of random deletes (used only if custom_delete is None)
      seed: Random seed for reproducibility
    """
    if seed is not None:
      random.seed(seed)

    if custom_insert is not None:
      created_numbers = custom_insert
      actual_create_count = len(custom_insert)
    else:
      created_numbers = random.sample(range(1, 100), create_count)
      actual_create_count = create_count

    if custom_delete is not None:
      deleted_numbers = custom_delete
      actual_delete_count = len(custom_delete)
    else:
      actual_delete_count = min(delete_count, len(created_numbers))
      deleted_numbers = random.sample(created_numbers, actual_delete_count)

    return cls(actual_create_count, actual_delete_count, created_numbers, deleted_numbers)

  def get_insert_commands(self) -> str:
    """Get insert commands only"""
    return '\n'.join(f'i {num}' for num in self.created_numbers)

  def get_full_commands(self) -> str:
    """Get insert + delete commands"""
    inserts = '\n'.join(f'i {num}' for num in self.created_numbers)
    deletes = '\n'.join(f'd {num}' for num in self.deleted_numbers)
    return f"{inserts}\n{deletes}"

  def print_summary(self):
    """Print test case summary"""
    print(f"Created numbers: {self.created_numbers}")
    print(f"Deleted numbers: {self.deleted_numbers}")
    print()


class TreeNode:
    """Generic tree node for visualization"""
    def __init__(self, keys: List[int]):
        self.keys = keys
        self.children: List[Optional['TreeNode']] = []

    def __repr__(self):
        return f"TreeNode({self.keys})"


class TreeParser:
  """Parse tree output strings into tree structures"""

  @staticmethod
  def parse_bst_avl(tree_str: str) -> Optional[TreeNode]:
    """
    Parse BST/AVL tree output like: <<< 3 > 5 < 7 >> 10 < 15 >>
    Returns a TreeNode structure
    """
    if not tree_str or tree_str.strip() == '':
      return None

    tree_str = tree_str.strip()
    return TreeParser._parse_bst_recursive(tree_str)

  @staticmethod
  def _parse_bst_recursive(s: str) -> Optional[TreeNode]:
    """Recursively parse BST/AVL format"""
    s = s.strip()
    if not s:
      return None

    # Remove outer < > if present
    if s.startswith('<') and s.endswith('>'):
      s = s[1:-1].strip()

    if not s:
      return None

    # Find the root value (the number not surrounded by < >)
    depth = 0
    parts = []
    current = []

    tokens = s.replace('<', ' < ').replace('>', ' > ').split()

    i = 0
    while i < len(tokens):
      token = tokens[i]
      if token == '<':
        depth += 1
        current.append(token)
      elif token == '>':
        depth -= 1
        current.append(token)
      else:
        if depth == 0:
          # This is a root value
          if current:
            parts.append(' '.join(current))
            current = []
          parts.append(token)
        else:
          current.append(token)
      i += 1

    if current:
      parts.append(' '.join(current))

    # Find root (should be a single number at depth 0)
    root_key = None
    left_str = None
    right_str = None

    for i, part in enumerate(parts):
      if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
        root_key = int(part)
        if i > 0:
          left_str = parts[i-1]
        if i < len(parts) - 1:
          right_str = parts[i+1]
        break

    if root_key is None:
      return None

    node = TreeNode([root_key])

    # Parse left and right children
    left_child = TreeParser._parse_bst_recursive(left_str) if left_str else None
    right_child = TreeParser._parse_bst_recursive(right_str) if right_str else None

    if left_child or right_child:
      node.children = [left_child, right_child]

    return node

  @staticmethod
  def parse_btree(tree_str: str) -> Optional[TreeNode]:
    """
    Parse B-Tree output like: << 1 2 > 5 < 10 > 20 < 30 >>
    Returns a TreeNode structure
    """
    if not tree_str or tree_str.strip() == '':
      return None

    tree_str = tree_str.strip()
    return TreeParser._parse_btree_recursive(tree_str)

  @staticmethod
  def _parse_btree_recursive(s: str) -> Optional[TreeNode]:
    """Recursively parse B-Tree format"""
    s = s.strip()
    if not s:
      return None

    # Remove outer < > if present
    if s.startswith('<') and s.endswith('>'):
      s = s[1:-1].strip()

    if not s:
      return None

    # Tokenize
    tokens = s.replace('<', ' < ').replace('>', ' > ').split()

    # Parse at depth 0
    depth = 0
    keys = []
    children_strs = []
    current_child = []

    for token in tokens:
      if token == '<':
        depth += 1
        current_child.append(token)
      elif token == '>':
        depth -= 1
        current_child.append(token)
        if depth == 0 and len(current_child) > 0:
          children_strs.append(' '.join(current_child))
          current_child = []
      else:
        if depth == 0:
          # This is a key at current level
          try:
            keys.append(int(token))
          except ValueError:
            pass
        else:
          current_child.append(token)

    if not keys:
      return None

    node = TreeNode(keys)

    # Parse children
    if children_strs:
      node.children = [TreeParser._parse_btree_recursive(child_str) for child_str in children_strs]
      node.children = [c for c in node.children if c is not None]

    return node


class TreeVisualizer:
    """Visualize tree structures"""

    @staticmethod
    def visualize_binary_tree(root: Optional[TreeNode]) -> List[str]:
      """
      Visualize binary tree horizontally with clear left/right structure
      Returns list of lines showing tree structure
      """
      if root is None:
        return ["(empty)"]

      def get_height(node):
        if not node:
          return 0
        left_h = get_height(node.children[0]) if node.children and len(node.children) > 0 else 0
        right_h = get_height(node.children[1]) if node.children and len(node.children) > 1 else 0
        return 1 + max(left_h, right_h)

      def build_tree_lines(node, width):
        """Build visual representation of tree"""
        if not node:
          return [], 0, 0, 0

        # Node value as string
        val_str = str(node.keys[0])
        val_len = len(val_str)

        left_child = node.children[0] if node.children and len(node.children) > 0 else None
        right_child = node.children[1] if node.children and len(node.children) > 1 else None

        # Recursively get left and right subtrees
        left_lines, left_pos, left_width, _ = build_tree_lines(left_child, width)
        right_lines, right_pos, right_width, _ = build_tree_lines(right_child, width)

        # Calculate positioning
        if not left_child and not right_child:
          # Leaf node
          return [val_str], val_len // 2, val_len, 1

        # Position for this node's value
        if left_child and right_child:
          # Has both children
          gap = 2
          total_width = left_width + gap + right_width

          # Node position is between the two children
          node_pos = left_width + gap // 2

          lines = []

          # Add value line
          padding_left = node_pos - val_len // 2
          lines.append(' ' * padding_left + val_str)

          # Add connector line
          left_connect = left_pos
          right_connect = left_width + gap + right_pos
          connector_line = [' '] * total_width

          # Draw branches
          for i in range(left_connect, right_connect + 1):
            if i == left_connect or i == right_connect:
              connector_line[i] = '┌' if i == left_connect else '┐'
            elif left_connect < i < right_connect:
              connector_line[i] = '─'

          lines.append(''.join(connector_line))
          lines.append(' ' * left_connect + '│' + ' ' * (right_connect - left_connect - 1) + '│')

          # Combine left and right subtree lines
          max_height = max(len(left_lines), len(right_lines))
          for i in range(max_height):
            left_part = left_lines[i] if i < len(left_lines) else ' ' * left_width
            right_part = right_lines[i] if i < len(right_lines) else ' ' * right_width

            # Pad left_part to left_width
            if len(left_part) < left_width:
              left_part += ' ' * (left_width - len(left_part))

            lines.append(left_part + ' ' * gap + right_part)

          return lines, node_pos, total_width, len(lines)

        elif left_child:
          # Only left child
          lines = []
          lines.append(' ' * left_pos + val_str)
          lines.append(' ' * left_pos + '│')
          lines.extend(left_lines)
          return lines, left_pos, left_width, len(lines)
        else:
          # Only right child
          lines = []
          lines.append(val_str)
          lines.append(' ' * (val_len // 2) + '│')
          right_lines_shifted = [' ' * (val_len // 2) + line for line in right_lines]
          lines.extend(right_lines_shifted)
          return lines, val_len // 2, val_len // 2 + right_width, len(lines)

      lines, _, _, _ = build_tree_lines(root, 80)
      return lines

    @staticmethod
    def visualize_btree(root: Optional[TreeNode]) -> List[str]:
      """
      Visualize B-Tree with horizontal layout showing all keys and children
      Returns list of lines
      """
      if root is None:
          return ["(empty)"]

      def build_btree_lines(node):
        """Build visual representation of B-Tree"""
        if not node:
          return [], 0, 0

        keys_str = " ".join(str(k) for k in node.keys)
        node_str = f"[{keys_str}]"
        node_width = len(node_str)

        if not node.children or all(c is None for c in node.children):
            return [node_str], node_width // 2, node_width

        children_data = []
        total_children_width = 0

        for child in node.children:
          if child:
            child_lines, child_pos, child_width = build_btree_lines(child)
            children_data.append((child_lines, child_pos, child_width))
            total_children_width += child_width
          else:
            children_data.append(([], 0, 0))

        gap = 2
        total_width = total_children_width + gap * (len(children_data) - 1)

        lines = []
        node_center = total_width // 2
        padding = max(0, node_center - node_width // 2)
        lines.append(' ' * padding + node_str)

        if children_data:
          connector_line = [' '] * total_width
          branch_line = [' '] * total_width

          current_x = 0
          child_positions = []

          for child_lines, child_pos, child_width in children_data:
            if child_width > 0:
              child_center = current_x + child_pos
              child_positions.append(child_center)
            current_x += child_width + gap

          if len(child_positions) > 1:
            for x in range(child_positions[0], child_positions[-1] + 1):
              connector_line[x] = '─'
            for pos in child_positions:
              connector_line[pos] = '┬'

            center_line = [' '] * total_width
            center_line[node_center] = '│'
            lines.append(''.join(center_line))
            lines.append(''.join(connector_line))

            for pos in child_positions:
              branch_line[pos] = '│'
            lines.append(''.join(branch_line))
          elif len(child_positions) == 1:
            # Single child
            center_line = [' '] * total_width
            center_line[node_center] = '│'
            lines.append(''.join(center_line))
            lines.append(''.join(center_line))

          # Combine children lines
          max_child_height = max((len(c[0]) for c in children_data), default=0)

          current_x = 0
          for row in range(max_child_height):
            line_parts = []
            for child_lines, child_pos, child_width in children_data:
              if row < len(child_lines):
                part = child_lines[row]
                # Pad to width
                if len(part) < child_width:
                  part += ' ' * (child_width - len(part))
              else:
                part = ' ' * child_width
              line_parts.append(part)
            lines.append((' ' * gap).join(line_parts))
        return lines, node_center, total_width

      lines, _, _ = build_btree_lines(root)
      return lines

    @staticmethod
    def print_tree(root: Optional[TreeNode], tree_type: str = "binary"):
      """Print tree visualization"""
      if root is None:
        print("(empty tree)")
        return

      if tree_type == "btree":
        lines = TreeVisualizer.visualize_btree(root)
      else:
        lines = TreeVisualizer.visualize_binary_tree(root)

      for line in lines:
        print(line)


class TreeTester:
    """Run tests on tree executables"""

    def __init__(self, submit_dir: str):
      self.submit_dir = Path(submit_dir)
      self.executables = self._load_executables()

    def _load_executables(self) -> dict:
      """Load tree executables from configuration constants"""
      execs = {}

      # Load BST executable
      if BST_EXECUTABLE:
        bst_path = self.submit_dir / BST_EXECUTABLE
        if bst_path.exists() and os.access(bst_path, os.X_OK):
          execs['BST'] = bst_path
        else:
          print(f"Warning: BST executable not found or not executable: {bst_path}")

      # Load AVL executable
      if AVL_EXECUTABLE:
        avl_path = self.submit_dir / AVL_EXECUTABLE
        if avl_path.exists() and os.access(avl_path, os.X_OK):
          execs['AVL'] = avl_path
        else:
          print(f"Warning: AVL executable not found or not executable: {avl_path}")

      # Load B-Tree executable
      if BTREE_EXECUTABLE:
        btree_path = self.submit_dir / BTREE_EXECUTABLE
        if btree_path.exists() and os.access(btree_path, os.X_OK):
          execs['BT'] = btree_path
        else:
          print(f"Warning: B-Tree executable not found or not executable: {btree_path}")

      return execs

    def run_test(self, executable: Path, commands: str) -> List[str]:
      """Run executable with commands and return output lines"""
      try:
        result = subprocess.run(
          [str(executable)],
          input=commands,
          capture_output=True,
          text=True,
          timeout=5
        )

        # Return stdout lines, filter out empty lines
        return [line for line in result.stdout.strip().split('\n') if line.strip()]
      except subprocess.TimeoutExpired:
        print(f"Error: {executable.name} timed out")
        return []
      except Exception as e:
        print(f"Error running {executable.name}: {e}")
        return []

    def test_tree(self, tree_name: str, test_case: TestCase):
      """Test a specific tree implementation"""
      if tree_name not in self.executables:
        print(f"Error: {tree_name} executable not found")
        return

      executable = self.executables[tree_name]
      is_btree = tree_name == 'BT'

      print(f"\n{'='*80}")
      print(f"Testing {tree_name}: {executable.name}")
      print(f"{'='*80}\n")

      if not SHOW_ALL_IN_ONE_TIME:
        input("Press Enter to continue...")

      # Test 1: Insert only
      print(f"{'─'*80}")
      print("Test 1: Insert Operations Only")
      print(f"{'─'*80}")

      insert_commands = test_case.get_insert_commands()
      insert_outputs = self.run_test(executable, insert_commands)

      if insert_outputs:
        if SKIP_PROGRESS:
          # Show only final state
          final_output = insert_outputs[-1]
          print(f"\nFinal tree structure (raw): {final_output}")
          print(f"\nVisualization:")

          if is_btree:
            tree = TreeParser.parse_btree(final_output)
            TreeVisualizer.print_tree(tree, "btree")
          else:
            tree = TreeParser.parse_bst_avl(final_output)
            TreeVisualizer.print_tree(tree, "binary")
        else:
          # Show every step
          commands = insert_commands.split('\n')
          for i, output in enumerate(insert_outputs):
            print(f"\n[Step {i+1}] After: {commands[i]}")
            print(f"Tree structure (raw): {output}")
            print("Visualization:")

            if is_btree:
              tree = TreeParser.parse_btree(output)
              TreeVisualizer.print_tree(tree, "btree")
            else:
              tree = TreeParser.parse_bst_avl(output)
              TreeVisualizer.print_tree(tree, "binary")
            print()

            # Wait for input between steps if interactive mode
            if not SHOW_ALL_IN_ONE_TIME and i < len(insert_outputs) - 1:
              input("Press Enter for next step...")
      else:
        print("No output received")

      if not SHOW_ALL_IN_ONE_TIME:
        input("\nPress Enter to continue to Test 2...")

      # Test 2: Insert + Delete
      print(f"\n{'─'*80}")
      print("Test 2: Insert + Delete Operations")
      print(f"{'─'*80}")

      full_commands = test_case.get_full_commands()
      full_outputs = self.run_test(executable, full_commands)

      if full_outputs:
        # Calculate how many outputs correspond to insert operations
        insert_count = len(test_case.created_numbers)
        delete_count = len(test_case.deleted_numbers)

        if SKIP_PROGRESS:
          final_output = full_outputs[-1]
          print(f"\nFinal tree structure (raw): {final_output}")
          print(f"\nVisualization:")

          if is_btree:
            tree = TreeParser.parse_btree(final_output)
            TreeVisualizer.print_tree(tree, "btree")
          else:
            tree = TreeParser.parse_bst_avl(final_output)
            TreeVisualizer.print_tree(tree, "binary")
        else:
          # Show initial state (after all inserts) and then delete steps
          commands = full_commands.split('\n')

          # Show state after all insertions (before any deletions)
          if len(full_outputs) > delete_count:
            initial_state_output = full_outputs[insert_count - 1]
            print(f"\n[Initial State] After all insertions:")
            print(f"Tree structure (raw): {initial_state_output}")
            print("Visualization:")

            if is_btree:
              tree = TreeParser.parse_btree(initial_state_output)
              TreeVisualizer.print_tree(tree, "btree")
            else:
              tree = TreeParser.parse_bst_avl(initial_state_output)
              TreeVisualizer.print_tree(tree, "binary")
            print()

            if not SHOW_ALL_IN_ONE_TIME:
              input("Press Enter to start deletions...")

          # Show only delete steps
          delete_step = 1
          for i in range(insert_count, len(full_outputs)):
            output = full_outputs[i]
            command = commands[i]

            print(f"\n[Delete Step {delete_step}] After: {command}")
            print(f"Tree structure (raw): {output}")
            print("Visualization:")

            if is_btree:
              tree = TreeParser.parse_btree(output)
              TreeVisualizer.print_tree(tree, "btree")
            else:
              tree = TreeParser.parse_bst_avl(output)
              TreeVisualizer.print_tree(tree, "binary")
            print()

            delete_step += 1

            if not SHOW_ALL_IN_ONE_TIME and i < len(full_outputs) - 1:
              input("Press Enter for next step...")
      else:
        print("No output received")

    def test_all(self, test_case: TestCase):
      """Test all tree implementations"""
      print("\n" + "="*80)
      print("TREE TESTING AND VISUALIZATION")
      print("="*80)

      test_case.print_summary()

      if not SHOW_ALL_IN_ONE_TIME:
        input("Press Enter to show tree...")

      for tree_name in ['BST', 'AVL', 'BT']:
        if tree_name in self.executables:
          self.test_tree(tree_name, test_case)

          if not SHOW_ALL_IN_ONE_TIME and tree_name != 'BT':
            input(f"\nPress Enter to continue to next tree type...")
        else:
          print(f"\nSkipping {tree_name}: executable not found")


def main():
  """Main function"""
  SUBMIT_DIR = Path(__file__).parent / "submit"

  # Show current configuration
  print("="*80)
  print("CONFIGURATION")
  print("="*80)
  print(f"SKIP_PROGRESS: {SKIP_PROGRESS}")
  print(f"  → {'Show only final results' if SKIP_PROGRESS else 'Show every step (i/d commands)'}")
  print(f"SHOW_ALL_IN_ONE_TIME: {SHOW_ALL_IN_ONE_TIME}")
  print(f"  → {'Display all at once' if SHOW_ALL_IN_ONE_TIME else 'Wait for Enter key between sections'}")

  # Display test case configuration
  if CUSTOM_INSERT_VALUES:
    print(f"\nTest case: Custom insert values ({len(CUSTOM_INSERT_VALUES)} values)")
    if CUSTOM_DELETE_VALUES:
      print(f"           Custom delete values ({len(CUSTOM_DELETE_VALUES)} values)")
    else:
      print(f"           Random delete from insert values")
  else:
    print(f"\nTest case: Random generation")
    print(f"           {RANDOM_CREATE_COUNT} insertions, {RANDOM_DELETE_COUNT} deletions")

  # Display executable configuration
  print(f"\nExecutables:")
  print(f"  BST:    {BST_EXECUTABLE or 'None (skip)'}")
  print(f"  AVL:    {AVL_EXECUTABLE or 'None (skip)'}")
  print(f"  B-Tree: {BTREE_EXECUTABLE or 'None (skip)'}")
  print("="*80)

  if not SHOW_ALL_IN_ONE_TIME:
    input("\nPress Enter to start...")

  # Check if submit directory exists
  if not SUBMIT_DIR.exists():
    print(f"Error: Submit directory not found: {SUBMIT_DIR}")
    sys.exit(1)

  # Generate test case using configuration constants
  test_case = TestCase.generate(
    custom_insert=CUSTOM_INSERT_VALUES,
    custom_delete=CUSTOM_DELETE_VALUES,
    create_count=RANDOM_CREATE_COUNT,
    delete_count=RANDOM_DELETE_COUNT
  )

  # Run tests
  tester = TreeTester(str(SUBMIT_DIR))
  tester.test_all(test_case)

  print("\n" + "="*80)
  print("Testing complete!")
  print("="*80)


if __name__ == "__main__":
  main()
