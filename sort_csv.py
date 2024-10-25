import csv
import argparse

parser = argparse.ArgumentParser(description="Sort the classification CSV file")
parser.add_argument('--input-file', help='File where the classifications are present', type=str)
parser.add_argument('--output-file', help='File where sorted classifications should be saved', type=str)
args = parser.parse_args()

if args.input_file is None or args.output_file is None:
    print("Please specify all required arguments (Input file and Output file names)")
    exit(1)

with open(args.input_file, 'r') as file:
    reader = csv.reader(file)
    sorted_rows = sorted(reader, key=lambda row:row[0])

with open(args.output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sorted_rows)