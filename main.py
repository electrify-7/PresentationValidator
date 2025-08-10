import sys
import os

def main():
    # Take the file either as argument or through CLI files.
    options = ["PPTX file", "Images", "Folder containing images"]
    choice = 0

    if os.name == "nt":
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        choice = input("Enter choice number: ")
        try:
            choice = options[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice, exiting...")
            sys.exit(1)
    else:
        from simple_term_menu import TerminalMenu
        terminal_menu = TerminalMenu(options, title="Select input type:")
        menu_index = terminal_menu.show()

        print("\n")
        if menu_index is None:
            print("No option selected, exiting...")
            sys.exit(1)

        choice = options[menu_index]

    if choice == "PPTX file":
        print("Downloading Dependencies...\n")
        import pptx_parser
        import image_captioning
        import json_converter
        import llm_query
        import console_print

        file_address = input("Enter the pptx file path: ").strip()
        if not file_address:
            print("No file path entered, exiting...")
            sys.exit(0)
        if not os.path.isfile(file_address):
            print("File does not exist, exiting...")
            sys.exit(0)
        print(f"Processing file...")
        print("\n" * 3)

        # Redirect to pptx_parser.py for handling parsing logic.
        if file_address.lower().endswith(".pptx"):
            collected_data = pptx_parser.parse(file_address)
            
            # Caption all the embedded images
            collected_data = image_captioning.caption(collected_data)
            #pprint.pprint(collected_data)

            # convert into Json and group slide content
            formatted_data = json_converter.format_dict(collected_data)
            #print(json.dumps(formatted_data, indent=2))
            if len(formatted_data) > 10:
                formatted = llm_query.preprocess_slides_with_llm(formatted_data)
            else:
                formatted = formatted_data

            # Do a LLM query (gemini or whatever api you're using. )
            response = llm_query.check_inconsistencies(formatted)
            #pprint.pprint(response)
            console_print.pretty_print(response)
    elif choice == "Folder containing images":
        if os.name == 'nt' and 'TESSERACT_CMD' in os.environ:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']

        import image_preprocessing
        import json_converter
        import llm_query
        import console_print

        file_address = input("Enter the folder's address: ").strip()
        if not file_address:
            print("No folder path entered, exiting...")
            sys.exit(1)
        print(f"Processing folder: {file_address}")

        if os.path.isdir(file_address):
            # Iteratively.
            print("Folder detected. Scanning for image files...")
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
            images = [
                os.path.join(file_address, f)
                for f in os.listdir(file_address)
                if f.lower().endswith(image_extensions)
            ]
            if not images:
                print("No supported image files found in the folder.")
                sys.exit(1)
            collected_data = image_preprocessing.process_folder(file_address)  
            formatted_data = json_converter.format_dict(collected_data) 

            if len(formatted_data) > 10:
                formatted = llm_query.preprocess_slides_with_llm(formatted_data)
            else:
                formatted = formatted_data
             
            response = llm_query.check_inconsistencies(formatted)
            console_print.pretty_print(response)     

    elif choice == "Images":
            if os.name == 'nt' and 'TESSERACT_CMD' in os.environ:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']            

            import image_preprocessing
            import json_converter
            import llm_query
            import console_print

            file_addresses = input("Enter the image file paths separated by space: ").strip().split()
            if not file_addresses:
                print("No image file paths entered, exiting...")
                sys.exit(0)

            combined_data = {
                'file': None,
                'slide_titles': [],
                'texts': [],
                'alt_texts': [],
                'tables': [],
                'images': [],
                'charts': [],
                'notes': [],
                'elements': []
            }

            slide_counter = 0
            for file_address in file_addresses:
                if os.path.isfile(file_address) and file_address.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    print(f"Processing image: {file_address}")
                    single_data = image_preprocessing.process_folder(file_address)

                    # if multiple images path given seperated then make sure sequentially slide number is counted.
                    for key in ['slide_titles', 'texts', 'images', 'elements']:
                        for entry in single_data[key]:
                            entry['slide'] = slide_counter
                        slide_counter += 1 if key == 'slide_titles' else 0

                    # Merge into one dict.
                    for k in combined_data:
                        combined_data[k].extend(single_data[k]) if isinstance(combined_data[k], list) else None
                else:
                    print(f"Invalid image file: {file_address}. Skipping...")

            collected_data = combined_data
            formatted_data = json_converter.format_dict(collected_data)  

            if len(formatted_data) > 10:
                formatted = llm_query.preprocess_slides_with_llm(formatted_data)
            else:
                formatted = formatted_data
            
            response = llm_query.check_inconsistencies(formatted)
            console_print.pretty_print(response)     
    else:
        print("Not supported. Please give either .pptx or a directory/image reference")
        sys.exit(0)


if __name__ == "__main__":
    main()