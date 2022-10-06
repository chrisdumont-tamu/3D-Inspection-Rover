import csv
import requests
import time

############################################################
# This section of code is used to upload data to DynamoDB

DYNAMODB_ENDPOINT = 'https://cnpu0bqb4i.execute-api.us-east-1.amazonaws.com/beta/items'

def upload_piece_data(identifier, aisle, stage, volume):
    data = {"id": identifier, "Aisle": aisle, "Stage": stage, "Volume":volume}
    reply = requests.put(DYNAMODB_ENDPOINT, json = data)
    return reply.status_code

def upload_folder_data(file_name):
    
    with open(file_name) as f:
        ## Variables
        line_index = 0
        id_index = 0
        stage_index = 2
        yiled_index = 7
        
        csv_reader = csv.reader(f, delimiter=',')
        
        for row in csv_reader:
            if line_index == 0:
            
                line_index += 1
                continue
            
            identifier = row[id_index]
            stage = row[stage_index]
            
            upload_piece_data(identifier, 0, stage, 0)
            print(identifier, 0, stage, 0)
        
            line_index += 1

def create_example_csv(file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header_row = ['id', 'Aisle', 'Stage', 'Volume']
        writer.writerow(header_row)
        
        for i in range(250):
            row = [f'tomato{i}', i, f'stage{i}', i]
            writer.writerow(row)

############################################################

## Constants
# FILE_NAME = 'example.csv'

# start = time.time()

# upload_folder_data(FILE_NAME)

# end = time.time()

# total_time = (end - start)

# print(f'\nTotal Execution Time: {total_time} S')

## Note: It took ~70S to upload 250 lines of data to Celeste's database




# Create a csv file for througput evaluation
EXAMPLE_FILE_NAME = 'example.csv'
create_example_csv(EXAMPLE_FILE_NAME)