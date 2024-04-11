import pandas as pd
import os
import data_process as dp

def get_character_lines(df, characters):
    # Filter DataFrame to select rows where 'character' matches any of the desired characters
    character_lines = []
    for character in characters:
        character_df = df[df['character'] == character]
        
        for i, row in character_df.iterrows():
            index = row.name
            current_line = row['line']
            
            # Retrieve the preceding line if available, else assign None
            preceding_line = df.loc[index - 1, 'line'] if index > 0 else None
            
            # Retrieve the succeeding line if available, else assign None
            succeeding_line = df.loc[index + 1, 'line'] if index < len(df) - 1 else None
            
            context = {'preceding_line': preceding_line, 'succeeding_line': succeeding_line}
            character_lines.append((current_line, context))
    
    return character_lines

def read_directory(directory, characters):
    all_lines = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            df = pd.read_csv(f)
            
            # Now you can perform operations on df, such as calling get_character_lines
            lines = get_character_lines(df, characters)
            
            # Extend the all_lines list with the lines from the current file
            all_lines.extend(lines)

    return all_lines
    


characters = ["TONY STARK", "TONY"]
data = read_directory('./data/raw', characters)

with open('./data/processed/tony-with-context.txt', 'w') as f:
    for line, context in data:
        clean_line = dp.clean_text(line)
        clean_prec_context = dp.clean_text(context['preceding_line']) if context['preceding_line'] != None else ''
        # clean_succ_context = dp.preprocess_data(context['succeeding_line']) if context['succeeding_line'] != None else ''

        f.write('[Context] ' + clean_prec_context + '\n')
        f.write('[PatternToLearn] ' + clean_line + '\n')
