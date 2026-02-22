# Unified Color Configuration for YOLOv11 Cheque Detection
# All colors are in BGR format (OpenCV standard)

# Standard color palette for all cheque field classes
CLASS_COLORS = {
    'Account_Number': (255, 0, 0),      # Blue
    'Amount': (0, 0, 255),              # Red
    'Bank_Name': (0, 140, 255),         # Orange
    'Date': (255, 0, 255),              # Magenta
    'IFSC': (0, 255, 255),              # Yellow
    'MICR': (128, 0, 128),              # Purple
    'Signature': (0, 128, 0)            # Dark Green
}

# Color labels for reference
COLOR_LABELS = {
    'Account_Number': 'Blue',
    'Amount': 'Red',
    'Bank_Name': 'Orange',
    'Date': 'Magenta',
    'IFSC': 'Yellow',
    'MICR': 'Purple',
    'Signature': 'Dark Green'
}

# Convert colors to format YOLOv11 expects (RGB tuples as list)
def get_color_palette(class_names):
    """
    Convert CLASS_COLORS dict to palette list matching class_names order
    
    Args:
        class_names: List of class names from dataset
        
    Returns:
        List of BGR tuples in same order as class_names
    """
    palette = []
    for class_name in class_names:
        color = CLASS_COLORS.get(class_name, (0, 255, 0))  # Default to Green
        palette.append(color)
    return palette


def print_color_guide():
    """Print a guide of all colors being used"""
    print("\n" + "="*50)
    print("CHEQUE FIELD COLOR GUIDE (BGR Format)")
    print("="*50)
    for field_name, color in CLASS_COLORS.items():
        label = COLOR_LABELS.get(field_name, "Unknown")
        print(f"{field_name:20} -> {label:15} BGR: {color}")
    print("="*50 + "\n")
