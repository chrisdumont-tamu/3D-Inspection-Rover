import './App.css';
import { useState, useEffect } from 'react';
//import Histogram from "react-chart-histogram";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
  } 

from 'chart.js';
import { Bar } from 'react-chartjs-2';
  
  ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
  );

function App() {
    const [tomatoes, setTomatoes] = useState([]);
    const labels = ['Tomatoes'];
    const options = {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Tomatoes by Volume (mL)',
          },
        },
      };
    
    const data = {
    labels,
    datasets: [
        {
        label: 'Green',
        data: [(tomatoes.filter(tomato => tomato.Stage === "green")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        }, 0)],
        backgroundColor: 'rgba(0, 160, 0, 5)',
        },
        {
        label: 'Breaker',
        data: [(tomatoes.filter(tomato => tomato.Stage === "breaker")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(153, 204, 0, 5)',
        },
        {
        label: 'Turning',
        data: [(tomatoes.filter(tomato => tomato.Stage === "turning")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(255, 179, 102, 5)',
        },
        {
        label: 'Pink',
        data: [(tomatoes.filter(tomato => tomato.Stage === "pink")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(255, 128, 128, 5)',
        },
        {
        label: 'Light Red',
        data: [(tomatoes.filter(tomato => tomato.Stage === "light red")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(255, 77, 77, 5)',
        },
        {
        label: 'Red',
        data: [(tomatoes.filter(tomato => tomato.Stage === "red")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(230, 57, 0, 5)',
        },
        {
        label: 'Deffective',
        data: [(tomatoes.filter(tomato => tomato.Stage === "deffective")).reduce((accumulator, object) => {
          return accumulator + object.Volume;
        },0 )],
        backgroundColor: 'rgba(77, 0, 0, 5)',
        },
    ],
    };

    //const options = {fillColor: '#F93208', strokeColor: '#F93208'};
    const fetchData = () => {
        fetch("https://cnpu0bqb4i.execute-api.us-east-1.amazonaws.com/beta/items")
            .then(response => {
                return response.json()
            })
            .then(data => {
                setTomatoes(data.Items)
            })
    }


    useEffect(() => {
        fetchData()
      }, [])

      return (
        <>
            <Bar options={options} data={data} />
        </>
    )
}

export default App;
