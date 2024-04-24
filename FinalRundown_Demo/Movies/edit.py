# Python program to split a text file into 10 equal parts

def split_file_into_parts(file_path):
    # Determine the base name for the output files
    base_name = file_path.rsplit('.', 1)[0]
    
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Calculate the length of each part
    part_length = len(content) // 10
    
    # Split the content into 10 parts and save them
    for i in range(10):
        start_index = i * part_length
        # For the last part, include any remaining content
        end_index = (i + 1) * part_length if i < 9 else None
        part_content = content[start_index:end_index]
        
        # Generate the filename for the part
        part_file_name = f"{base_name}_{i+1}.txt"
        
        # Write the part to a new file
        with open(part_file_name, 'w', encoding='utf-8') as part_file:
            part_file.write(part_content)

# Assuming the path to the file is provided, for example:
file_path = "Barbie.txt"
split_file_into_parts(file_path)
