import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
//configure the project to be used with AWS
import * as AWS from 'aws-sdk';
//import DWChart from "react-datawrapper-chart";



const configuration = {
  region: 'us-east-1',
  secretAccessKey: 't/k7TDOqf4vSD3I8gR6AOgyDTp1R56ArWSF0A6AY',
  accessKeyId: 'AKIAWZ2XQVV2TWHSYQE7',
}

AWS.config.update(configuration);


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

