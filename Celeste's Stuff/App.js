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
            text: 'Tomatoes by Count',
          },
        },
      };
    
    const data = {
    labels,
    datasets: [
        {
        label: 'Green',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Green")).length],
        backgroundColor: 'rgba(0, 160, 0, 5)',
        },
        {
        label: 'Breaker',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Breaker")).length],
        backgroundColor: 'rgba(153, 204, 0, 5)',
        },
        {
        label: 'Turning',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Turning")).length],
        backgroundColor: 'rgba(255, 179, 102, 5)',
        },
        {
        label: 'Pink',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Pink")).length],
        backgroundColor: 'rgba(255, 128, 128, 5)',
        },
        {
        label: 'Light Red',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Light Red")).length],
        backgroundColor: 'rgba(255, 77, 77, 5)',
        },
        {
        label: 'Red',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Red")).length],
        backgroundColor: 'rgba(230, 57, 0, 5)',
        },
        {
        label: 'Deffective',
        data: [(tomatoes.filter(tomato => tomato.Stage === "Deffective")).length],
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
