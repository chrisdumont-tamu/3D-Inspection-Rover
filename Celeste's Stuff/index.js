//intialize of the react which is coming from react
import React from 'react'; //entry point to the react library
import ReactDOM from 'react-dom/client'; //package provides client-specific methods used for initializing an app on the client
import './index.css'; //imports the css file into the JavaScript file
import App from './App'; //generates App.js
import reportWebVitals from './reportWebVitals'; //measures the performance of the app
//configure the project to be used with AWS
import * as AWS from 'aws-sdk';
//import DWChart from "react-datawrapper-chart";


//Now I set the AWS region for the JS code
const configuration = {
  region: 'us-east-1',
  secretAccessKey: 't/k7TDOqf4vSD3I8gR6AOgyDTp1R56ArWSF0A6AY',
  accessKeyId: 'AKIAWZ2XQVV2TWHSYQE7',
}

AWS.config.update(configuration); //set the global region setting


const root = ReactDOM.createRoot(document.getElementById('root')); //render react element, passes the DOM element
root.render( //pass react element
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals(); //gives the performance of app

