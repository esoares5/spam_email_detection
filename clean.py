import pandas as pd

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        cleaned_df = df.dropna()
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned CSV saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

input_csv = 'messages.csv'
output_csv = 'cleaned_messages.csv'

clean_csv(input_csv, output_csv)
