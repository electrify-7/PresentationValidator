import sys
import image_preprocessing
import os

def main():
    # Take the file either as argument or through CLI files.
    options = ["PPTX file", "Image file", "Folder containing images"]
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

            # Do a LLM query (gemini or whatever api you're using. )
            response = llm_query.check_inconsistencies(formatted_data)
            #pprint.pprint(response)
            #print(json.dumps(response, indent=2))
            # response = {
            #     "inconsistencies": [
            #         {
            #             "id": "inc-001",
            #             "type": "numeric_conflict",
            #             "slides": [
            #                 1,
            #                 2
            #             ],
            #             "statements": [
            #                 {
            #                     "slide": 1,
            #                     "text": "$2M Saved in lost productivity hours"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "text": "$3M saved in lost productivity hours annually"
            #                 }
            #             ],
            #             "evidence": [
            #                 {
            #                     "slide": 1,
            #                     "excerpt": "$2M"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "excerpt": "$3M saved in lost productivity hours annually"
            #                 }
            #             ],
            #             "explanation": "Slide 1 claims $2M saved in lost productivity hours. While not explicitly stated as 'annually', similar metrics in business presentations often imply an annual figure. Slide 2 explicitly states '$3M saved in lost productivity hours annually'. Assuming Slide 1 also refers to annual savings, these two values conflict.",
            #             "calculation_check": {
            #                 "parsed_values": [
            #                     {
            #                         "slide": 1,
            #                         "original": "$2M",
            #                         "value": 2000000,
            #                         "unit": "USD"
            #                     },
            #                     {
            #                         "slide": 2,
            #                         "original": "$3M",
            #                         "value": 3000000,
            #                         "unit": "USD"
            #                     }
            #                 ],
            #                 "derived_formula": None,
            #                 "expected": None,
            #                 "actual": None,
            #                 "difference": 1000000,
            #                 "relative_difference": 0.333
            #             },
            #             "conflict_kind": "value_discrepancy",
            #             "severity": "medium",
            #             "confidence": 0.7,
            #             "suggested_fix": "Clarify the precise period for the $2M saving on Slide 1, or reconcile the figures to present a consistent productivity saving value."
            #         },
            #         {
            #             "id": "inc-002",
            #             "type": "numeric_conflict",
            #             "slides": [
            #                 1,
            #                 2
            #             ],
            #             "statements": [
            #                 {
            #                     "slide": 1,
            #                     "text": "15 mins Saved per slide created"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "text": "20 mins saved per slide created"
            #                 }
            #             ],
            #             "evidence": [
            #                 {
            #                     "slide": 1,
            #                     "excerpt": "15 mins Saved per slide created"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "excerpt": "20 mins saved per slide created"
            #                 }
            #             ],
            #             "explanation": "Slide 1 states that '15 mins Saved per slide created', whereas Slide 2 states '20 mins saved per slide created'. These are contradictory values for the same efficiency metric.",
            #             "calculation_check": {
            #                 "parsed_values": [
            #                     {
            #                         "slide": 1,
            #                         "original": "15 mins",
            #                         "value": 15,
            #                         "unit": "minutes"
            #                     },
            #                     {
            #                         "slide": 2,
            #                         "original": "20 mins",
            #                         "value": 20,
            #                         "unit": "minutes"
            #                     }
            #                 ],
            #                 "derived_formula": None,
            #                 "expected": None,
            #                 "actual": None,
            #                 "difference": 5,
            #                 "relative_difference": 0.25
            #             },
            #             "conflict_kind": "value_discrepancy",
            #             "severity": "high",
            #             "confidence": 0.95,
            #             "suggested_fix": "Determine the correct time saved per slide and update both slides to reflect the accurate figure."
            #         },
            #         {
            #             "id": "inc-003",
            #             "type": "numeric_conflict",
            #             "slides": [
            #                 1,
            #                 2
            #             ],
            #             "statements": [
            #                 {
            #                     "slide": 1,
            #                     "text": "Case Study \u2013 Noogat helps consultants make decks 2x faster using AI"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "text": "3x faster deck creation speed"
            #                 }
            #             ],
            #             "evidence": [
            #                 {
            #                     "slide": 1,
            #                     "excerpt": "2x faster"
            #                 },
            #                 {
            #                     "slide": 2,
            #                     "excerpt": "3x faster deck creation speed"
            #                 }
            #             ],
            #             "explanation": "The title of Slide 1 claims Noogat helps consultants make decks '2x faster', while Slide 2 states '3x faster deck creation speed'. These multipliers for speed improvement are contradictory.",
            #             "calculation_check": {
            #                 "parsed_values": [
            #                     {
            #                         "slide": 1,
            #                         "original": "2x faster",
            #                         "value": 2,
            #                         "unit": "multiplier"
            #                     },
            #                     {
            #                         "slide": 2,
            #                         "original": "3x faster",
            #                         "value": 3,
            #                         "unit": "multiplier"
            #                     }
            #                 ],
            #                 "derived_formula": None,
            #                 "expected": None,
            #                 "actual": None,
            #                 "difference": 1,
            #                 "relative_difference": 0.333
            #             },
            #             "conflict_kind": "value_discrepancy",
            #             "severity": "high",
            #             "confidence": 0.95,
            #             "suggested_fix": "Confirm the actual improvement in deck creation speed (2x or 3x) and ensure consistency across all slides."
            #         },
            #         {
            #             "id": "inc-004",
            #             "type": "impossible_calculation",
            #             "slides": [
            #                 3
            #             ],
            #             "statements": [
            #                 {
            #                     "slide": 3,
            #                     "text": "Noogat: 50 Hours Saved Per Consultant Monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "text": "This saves an estimated 10 hours per consultant monthly."
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "text": "This saves an estimated 12 hours per consultant monthly."
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "text": "This saves an estimated 8 hours per consultant monthly."
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "text": "This saves an estimated 6 hours per consultant monthly."
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "text": "This saves an estimated 4 hours per consultant monthly."
            #                 }
            #             ],
            #             "evidence": [
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "Noogat: 50 Hours Saved Per Consultant Monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "10 hours per consultant monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "12 hours per consultant monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "8 hours per consultant monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "6 hours per consultant monthly"
            #                 },
            #                 {
            #                     "slide": 3,
            #                     "excerpt": "4 hours per consultant monthly"
            #                 }
            #             ],
            #             "explanation": "Slide 3 claims a total of '50 Hours Saved Per Consultant Monthly'. However, the sum of the detailed breakdown of time-saving areas on the same slide (10 + 12 + 8 + 6 + 4 hours) equals 40 hours. The stated total does not match the sum of its components.",
            #             "calculation_check": {
            #                 "parsed_values": [
            #                     {
            #                         "slide": 3,
            #                         "original": "50 Hours Saved Per Consultant Monthly",
            #                         "value": 50,
            #                         "unit": "hours"
            #                     },
            #                     {
            #                         "slide": 3,
            #                         "original": "10 hours",
            #                         "value": 10,
            #                         "unit": "hours"
            #                     },
            #                     {
            #                         "slide": 3,
            #                         "original": "12 hours",
            #                         "value": 12,
            #                         "unit": "hours"
            #                     },
            #                     {
            #                         "slide": 3,
            #                         "original": "8 hours",
            #                         "value": 8,
            #                         "unit": "hours"
            #                     },
            #                     {
            #                         "slide": 3,
            #                         "original": "6 hours",
            #                         "value": 6,
            #                         "unit": "hours"
            #                     },
            #                     {
            #                         "slide": 3,
            #                         "original": "4 hours",
            #                         "value": 4,
            #                         "unit": "hours"
            #                     }
            #                 ],
            #                 "derived_formula": "Sum of individual time savings (10+12+8+6+4) should equal the stated total.",
            #                 "expected": 50,
            #                 "actual": 40,
            #                 "difference": 10,
            #                 "relative_difference": 0.2
            #             },
            #             "conflict_kind": "aggregation_mismatch",
            #             "severity": "high",
            #             "confidence": 0.95,
            #             "suggested_fix": "Correct the total 'Hours Saved Per Consultant Monthly' on Slide 3 to reflect the accurate sum of its components (40 hours), or adjust the components to sum to 50 hours."
            #         }
            #     ],
            #     "summary": "4 inconsistencies found, including numeric conflicts in productivity savings, time saved per slide, and deck creation speed, along with an impossible calculation regarding total hours saved versus component breakdown.",
            #     "analysis_notes": ""
            # }
            console_print.pretty_print(response)
    elif choice == "Folder containing images":

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
            for image in images:
                #print(f"Processing image: {image}")
                temp_image = image_preprocessing.preprocess_image(image)  
                blocks = image_preprocessing.segment_layout(temp_image)          


    elif choice == "Images":
        file_addresses = input("Enter the image file paths separated by space: ").strip().split()
        if not file_addresses:
            print("No image file paths entered, exiting...")
            sys.exit(0)

        for file_address in file_addresses:
            if os.path.isfile(file_address) and file_address.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image
            else:
                print(f"Invalid image file: {file_address}. Skipping...")
            
    else:
        print("Not supported. Please give either .pptx or a directory/image reference")
        sys.exit(0)


if __name__ == "__main__":
    main()