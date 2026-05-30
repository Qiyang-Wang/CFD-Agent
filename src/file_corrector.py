import config
import shutil
import os
import file_writer

from qa_modules import QA_NoContext_deepseek_V3,QA_NoContext_deepseek_R1
import json
import re
import random
from pathlib import Path

def select_random_items(a, number):
    # Filter out key-value pairs where the value, when converted to a string, exceeds 10000 in length
    filtered_a = {k: v for k, v in a.items() if len(str(v)) <= 10000}
    
    # Check if the number of keys in the processed dictionary is greater than 5
    if len(filtered_a) > number:
        # Randomly select 5 keys
        selected_keys = random.sample(list(filtered_a.keys()), number)
        # Build a new dictionary
        return {key: filtered_a[key] for key in selected_keys}
    else:
        return filtered_a
        
    
def dict_to_json_string(data):
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    return json_str

def create_OF_case_json(source_dir):
    """
    Save the contents of OpenFOAM case directory (0/, constant/, system/) to a JSON document
    
    Args:
    source_dir - Path to the case root directory (folder containing 0/, constant/, system/)
    output_json - Path to the output JSON file
    
    Returns:
    None, results will be written to the specified JSON file
    """
    file_data = {}
    
    # Three target directories to process
    target_dirs = ["0", "constant", "system"]
    
    for dir_name in target_dirs:
        dir_path = os.path.join(source_dir, dir_name)
        
        # Skip if directory does not exist
        if not os.path.isdir(dir_path):
            continue
            
        # Iterate through entries in the directory
        for entry in os.listdir(dir_path):
            entry_path = os.path.join(dir_path, entry)
            
            # Only process files (ignore subdirectories)
            if os.path.isfile(entry_path):
                # Build relative path to source directory (using POSIX style path)
                relative_path = os.path.join(dir_name, entry).replace("\\", "/")
                
                try:
                    # Read file content
                    with open(entry_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_data[relative_path] = content
                except Exception as e:
                    print(f"Unable to read file {entry_path}: {str(e)}")
                    continue

    return dict_to_json_string(file_data)


def list_case_file(case_path):
    target_folders = ['0', 'system', 'constant']
    file_list = []
    for folder in target_folders:
        folder_path = os.path.join(case_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            if os.path.isfile(entry_path):
                # Build relative path and unify to Linux style path separator
                rel_path = f"{folder}/{entry}"
                file_list.append(rel_path)
    return file_list

def identify_error_to_add_new_file(running_error):

    analyze_error_to_add_new_file = f'''{config.general_prompts}. OpenFOAM File Requirement Analyzer
        Analyze the runtime error {running_error} to:

        1. Check if it contains the exact phrase "cannot find file"
        2. If present:
        a) Identify the missing file path from the error message
        b) Format the response as 0/..., system/..., or constant/...
        3. If absent/irrelevant: Respond with no

1. Respond ONLY with a filename (format: 0/xx, system/xx, or constant/xx) if required
2. Respond ONLY with 'no' if no file needed
3. Strict formatting requirements:
- No special characters: (), '', ", `
- No markdown/formatting symbols
- No whitespace, line breaks, or indentation
- No explanations or extra text
Your response must be exactly one of:
a) A directory-path formatted string from allowed locations
b) The lowercase string 'no'
Examples of valid responses:
system/fvSchemes
constant/g
no
'''

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(analyze_error_to_add_new_file)

    pure_response = file_writer.extract_pure_response(answer)

    return pure_response

def read_files_to_dict(base_dir):
    file_dict = {}
    # List of directories to traverse
    target_dirs = ['0', 'system', 'constant']
    
    for dir_name in target_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        # Skip if directory does not exist
        if not os.path.isdir(dir_path):
            continue
        # Iterate through all entries in the directory
        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)
            # Only process files, ignore subdirectories
            if os.path.isfile(full_path):
                # Build relative path as key (relative to base_dir)
                rel_path = os.path.join(dir_name, entry)
                # Read file content
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        file_dict[rel_path] = f.read()
                except Exception as e:
                    file_dict[rel_path] = f"<Error reading file: {str(e)}>"
    
    return file_dict

def add_new_file(file_name):

    print(f"adding new file: {file_name}")

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    other_case_file_content = read_files_to_dict(config.OUTPUT_PATH)

    add_new_file_prompt = f'''
    A new case file {file_name} must be add to the OpenFOAM case dir. The file contents of other case files are: {other_case_file_content}. Please respond the file contents for the new file which can make this case run correctly with other case files. Ensure the dimension is correct if the dimension shows in the file content.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or  ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''

    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(add_new_file_prompt)

    try:
        file_writer.write_field_to_file(answer,file_path)
        print(f"write the file {file_name}")
    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed the field file write operation
        file_write_successful = True

def identify_file_name_from_error(running_error):

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error { {running_error} } to identify the file requires revision. The result must be one of the following files: { {case_files} }. You response must only include the case name.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    answer = answer.strip()

    # file_for_revision = file_writer.extract_pure_response(answer)

    return answer

# Parameter target_file is the file name identified from running_error that needs to be modified
# Find by solver
def find_reference_files_by_solver(target_file):
    case_solver = config.case_solver
    turbulence_model = config.case_turbulece_model
    # Return content
    target_file_reference = {}

    solver_type = None

    file_number = 0

    for key, value in config.OF_case_data_dict.items():
        if case_solver in key:
            config_files = value['configuration_files']
            if target_file in config_files.keys():
                # print(config_files.keys())
                new_file_key = f'sample_file_{file_number}'
                file_number += 1
                target_file_reference[new_file_key] = config_files[target_file]

    # If the above result is 0, then look at a higher level, first find the solver type, such as compressible
    if file_number == 0:
        for key, value in config.OF_case_data_dict.items():
            if case_solver in key:
                path_split = key.split('/')
                solver_type = path_split[0]
                break

    # Find target_file under solver type
    if solver_type is not None:
        for key, value in config.OF_case_data_dict.items():
                path_split = key.split('/')
                if solver_type == path_split[0]:
                    config_files = value['configuration_files']
                    if target_file in config_files.keys():
                        # print(config_files.keys())
                        new_file_key = f'sample_file_{file_number}'
                        file_number += 1
                        target_file_reference[key] = config_files[target_file]

    several_target_file_reference = select_random_items(target_file_reference, 3)

    return dict_to_json_string(several_target_file_reference)

def analyze_running_error_with_all_case_file_content(running_error):

    all_case_file_content = create_OF_case_json(config.OUTPUT_PATH)

    file_content = None

    case_files = list_case_file(config.OUTPUT_PATH)

    case_files = dict_to_json_string(case_files)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error { {running_error} } to identify the root cause and which case file needs to be revised to fix the runtime error. The OpenFOAM case files are given as json-format string as { {all_case_file_content} }. 
    Your response must be a json format string with following keys and values:
    - a 'wrong_file' key, and its value the file name which will be revised to fix the error. This file name must be one of the case files { {case_files} }.
    - a 'advices_for_revision' key, and its value provide a step-by-step fix of the 'wrong_file' to fix the error. Ensure the advice addresses the error’s technical cause (e.g., CFL violation, invalid discretization scheme, missing required keyword). The advice must be a string.

    In your JSON response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    dict_answer = json.loads(answer)

    advices_for_revision = dict_answer['advices_for_revision']
    wrong_file = dict_answer['wrong_file']

    return [wrong_file,advices_for_revision]

def analyze_running_error_2(running_error, file_name):

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error {running_error} to identify the root cause. Give advice on correcting the file {file_name} with the file contents as {file_content}.

    In your response: Provide a step-by-step fix (e.g., adjust endTime, modify tolerance in solver settings, correct boundary type in U). Ensure the advice addresses the error’s technical cause (e.g., CFL violation, invalid discretization scheme, missing required keyword). The advice must be a string.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    advices_for_revision = answer

    return advices_for_revision

def analyze_error_repetition(error_history):
    answer = 'no'
    if(len(error_history)>=3):
        error_minus_1 = error_history[-1]
        error_minus_2 = error_history[-2]
        error_minus_3 = error_history[-3]

        analyze_running_error_repetition_prompt = f'''
        Analyze the following three error histories to identify whether the error have repetitively shown three time. Error 1: {error_minus_1}. Error 2: {error_minus_2}. Error 3: {error_minus_3}. If the error have repetitively shown three times, respond 'yes'; otherwise, respond 'no'. You must only respond 'yes' or 'no'.
        '''

        qa = QA_NoContext_deepseek_R1()

        answer = qa.ask(analyze_running_error_repetition_prompt)

    answer = answer.strip()

    if answer.lower() == 'yes':
        return True
    else:
        return False

def rewrite_file(file_name, reference_files):
    print(f"rewriting {file_name}")

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    correct_file_prompt = f'''{config.general_prompts}
    Please rewrite the {file_name} file for the OpenFOAM case. The original file content is: {file_content}. You can reference these files from OpenFOAM tutorial {reference_files} for formating and key values. Ensure the dimension is correct if the dimension shows in the file content.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or  ```)
    - Extra comments or explanations
    '''

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(correct_file_prompt)

    answer = file_writer.extract_pure_response(answer)

    try:
        file_writer.write_field_to_file(answer,file_path)
        print(f"write the file {file_name}")

    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed the field file write operation
        file_write_successful = True

def detect_dimension_error(running_error):
    
    detect_dimension_error = f'''{config.general_prompts}\nAnalyze the OpenFOAM runtime error {running_error} to determine if it explicitly indicates a dimensional inequality. Look for patterns such as:

    - Imbalanced dimension comparisons (e.g., [dim1] != [dim2], a != b where a and b - are dimensions).
    - Keywords like dimensions, incompatible dimensions, or dimension mismatch.
    - Explicit dimension lists (e.g., [0 0 0 -1 0 0 0] != [0 0 0 -2 0 0 0])
    If the error directly references dimensional inequality (as defined above), respond 'yes'. If not, or if the error is unrelated (e.g., syntax, segmentation faults, solver crashes), respond 'no'. Only reply with 'yes' or 'no'.'''

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(detect_dimension_error)

    answer = file_writer.extract_pure_response(answer)

    if answer.strip().lower() == 'yes':
        return True
    else:
        return False

def strongly_correct_all_dimension_with_reference_files():

    print(f"strongly correcting all field file dimension")

    case_0_folder = f'{config.OUTPUT_PATH}/0'

    folder_path = Path(case_0_folder)

    for file_path in folder_path.iterdir():
        file_content = None
        file_name = f'0/{file_path.name}'

        reference_files = find_reference_files_by_solver(file_name)

        if file_path.is_file():
            field_file = f'{case_0_folder}/{file_path.name}'

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            a = 1

        correct_dimension_prompt =   f'''{config.general_prompts}
        Please check the dimension of the OpenFOAM field file {file_path.name} and correct it according to the dimensions of the reference file contents. The original file content is: {file_content}. The reference file contents are: {reference_files}. You must not revise any other contents of the original file except for the dimension.

        In your response: Absolutely AVOID any elements including but not limited to:
        - Markdown code block markers (``` or  ```)
        - Extra comments or explanations
        - Unnecessary empty lines or indentation
        '''

        qa = QA_NoContext_deepseek_V3()

        answer = qa.ask(correct_dimension_prompt)

        answer = file_writer.extract_pure_response(answer)

        try:
            file_writer.write_field_to_file(answer,file_path)
            print(f"write the file 0/{file_path.name}")

        except Exception as e:
            print(f"Errors occur during write_field_to_file: {e}")
        else: # Successfully executed the field file write operation
            file_write_successful = True

def analyze_running_error_with_reference_files(running_error, file_name,early_revision_advice, reference_files):

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error [[[ {running_error} ]]] to identify the root cause. Give advice on correcting the file { {file_name} } with the file contents as [[[ {file_content} ]]]. The revision must not alter the file to voilate these initial and boundary conditions in the paper [[[ {config.case_ic_bc_from_paper} ]]]. You can refer to these files from OpenFOAM tutorial [[[ {reference_files} ]]] to improve the correction advice.

    In your response: Provide a step-by-step fix. Ensure the advice addresses the error's technical cause. The advice must be a string.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    advices_for_revision = answer

    return advices_for_revision

def single_file_corrector2(file_name, advices_for_revision, reference_files):
    print(f"correcting {file_name}")

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    correct_file_prompt = f'''{config.general_prompts} Correct the OpenFOAM case file.
    Please correct the { {file_name} } file with file contents as { {file_content} } to strictly adhere to the following correction advice { {advices_for_revision} }. Ensure the dimension in [] is correct if the dimension shows in the file content. You must not change any other contents of the file except for the correction advice or dimension in [].

    In your final response after "Here is my response:", absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or  ```)
    - Extra comments or explanations
    '''

    #     # You can reference these files from OpenFOAM tutorial { {reference_files} } for formatting.

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(correct_file_prompt)

    answer = file_writer.extract_pure_response(answer)

    try:
        file_writer.write_field_to_file(answer,file_path)
        print(f"write the file {file_name}")

    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed the field file write operation
        file_write_successful = True

def ensure_all_field_file_dimensions():

    print(f"ensuring all field file dimension")

    case_0_folder = f'{config.OUTPUT_PATH}/0'

    folder_path = Path(case_0_folder)

    for file_path in folder_path.iterdir():
        file_content = None
        if file_path.is_file():
            field_file = f'{case_0_folder}/{file_path.name}'

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            a = 1

        # reference_files = find_reference_files_by_solver(field_file)

        correct_dimension_prompt =   f'''{config.general_prompts}
        Please check the dimension of the OpenFOAM field file {file_path.name} and correct it if the dimension is incorrect. The file content is: {file_content}.

        In your response: Absolutely AVOID any elements including but not limited to:
        - Markdown code block markers (``` or  ```)
        - Extra comments or explanations
        - Unnecessary empty lines or indentation
        '''

        qa = QA_NoContext_deepseek_V3()

        answer = qa.ask(correct_dimension_prompt)

        answer = file_writer.extract_pure_response(answer)

        try:
            file_writer.write_field_to_file(answer,file_path)
            print(f"write the file 0/{file_path.name}")

        except Exception as e:
            print(f"Errors occur during write_field_to_file: {e}")
        else: # Successfully executed the field file write operation
            file_write_successful = True
