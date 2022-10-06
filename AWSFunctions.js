import * as AWS from 'aws-sdk';
//This code fetches data from DynamoDB and prints it in the console
const ddb = new AWS.DynamoDB.DocumentClient({
    region: 'us-east-1',
    secretAccessKey: 't/k7TDOqf4vSD3I8gR6AOgyDTp1R56ArWSF0A6AY',
    accessKeyId: 'AKIAWZ2XQVV2TWHSYQE7',
    apiVersion: 'latest',
});
 

//make this function available to the rest of the code

export const fetchData = (tableName) => {
    var params = {
        TableName: tableName,
    };

    ddb.scan(params, function(err, data) {
        if(err) {
            console.error(err);
        }
        else {
            console.log(data);
        }
    });
} 
/*
export const putData = (tableName , data) => {
    var params = {
        TableName: tableName,
        Item: data
    }
    
    docClient.put(params, function (err, data) {
        if (err) {
            console.log('Error', err)
        } else {
            console.log('Success', data)
        }
    })
}*/