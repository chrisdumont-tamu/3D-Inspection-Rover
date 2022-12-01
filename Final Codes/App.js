import './App.css'; //creates the App.cs file
import { useState, useEffect } from 'react'; //import usestate and useeffect
//import Histogram from "react-chart-histogram";
import { 
    Chart as ChartJS, //import chart library to make bar chart
    CategoryScale,  //This sets to the default scales
    LinearScale, //This sets to the default scales
    BarElement, //This sets the Bar chart's element
    Title, // Here we define text to draw at the top of the chart
    Tooltip, //Tooltip engulfs text alignment, intersection, position, background color, etc.
    Legend,//Legend will allow us to separate the days from each other later
  } 

from 'chart.js'; 
import { Bar } from 'react-chartjs-2'; //imports bar graph from package
  
  ChartJS.register( 
    CategoryScale, //This sets to the default scales
    LinearScale, //This sets to the default scales
    BarElement, //This sets the Bar chart's element
    Title, // Here we define text to draw at the top of the chart
    Tooltip, //Tooltip engulfs text alignment, intersection, position, background color, etc.
    Legend //Legend will allow us to separate the days from each other later
  );

function App() {
    const [tomatoes, setTomatoes] = useState([]);
    const labels = ['green', 'breaker', 'turning', 'pink', 'light red', 'red', 'defective']; //these are the labels for the x-axis sprted by tomato stage
    const options = { //set options for bar chart
        responsive: true, //enable responsiveness
        plugins: {
          legend: { 
            position: 'top', //adjust location of the legend to the top
          },
          title: {
            display: true,
            text: 'Tomatoes by Volume (L)', //Title of graph
          },
        },
      };
    
    const date = new Date(); //creates a variable to search for date
    const t_date = (date.getMonth() + 1) + "-" + date.getDate() + "-" + date.getFullYear(); //search for todays date
    const y_date = (date.getMonth() + 1) + "-" + (date.getDate() - 1) + "-" + date.getFullYear(); //search for yesterdays date

    const data = {
    labels,
    datasets: [
        {
        label: 'Today', //This will label all the data we call below as 'Today' in the graph
        data: [
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "green")).reduce((accumulator, object) => { //all data in the database that is green and from today is gathered and count all instances of it
            return accumulator + object.Volume;  //returns the total amount of green from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "breaker")).reduce((accumulator, object) => { //all data in the database that is breaker and from today is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of breaker from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "turning")).reduce((accumulator, object) => { //all data in the database that is turning and from today is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of turning from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "pink")).reduce((accumulator, object) => { //all data in the database that is pink and from today is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of pink from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "light red")).reduce((accumulator, object) => { //all data in the database that is light red and from today is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of light red from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "red")).reduce((accumulator, object) => { //all data in the database that is red and from today is gathered and count all instances of it
              return accumulator + object.Volume; //returns the total amount of red from today 
            }, 0),
            (tomatoes.filter(tomato => tomato.Timestamp === t_date && tomato.Stage === "defective")).reduce((accumulator, object) => { //all data in the database that is deffective and from today is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of deffective from today 
            }, 0),
        ],
        backgroundColor: 'rgba(255, 99, 132, .5)', //displays all the tomatoes from today as pink
        },
        {
            label: 'Yesterday', //This will label all the data we call below as 'Yesterday' in the graph
            data: [
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "green")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                return accumulator + object.Volume; //returns the total amount of green from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "breaker")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of breaker from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "turning")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of turning from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "pink")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of pink from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "light red")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of light red from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "red")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of red from today 
                }, 0),
                (tomatoes.filter(tomato => tomato.Timestamp === y_date && tomato.Stage === "defective")).reduce((accumulator, object) => { //all data in the database that is green and from yesterday is gathered and count all instances of it
                    return accumulator + object.Volume; //returns the total amount of deffective from today 
                }, 0),
            ],
            backgroundColor: 'rgba(53, 162, 235, .5)', //displays all of the tomatoes from yesterday as blue
            },
    ],
    };

    //const options = {fillColor: '#F93208', strokeColor: '#F93208'};
    const fetchData = () => {
        fetch("https://cnpu0bqb4i.execute-api.us-east-1.amazonaws.com/beta/items") //fetching the api gateway 
            .then(response => {
                return response.json() //parses JSON response into native JavaScript objects
            })
            .then(data => {
                setTomatoes(data.Items) //sets tomato as the data read in from database
            })
    }


    useEffect(() => { //define useeffect hook
        fetchData() //pulls data 
      }, [])

      return ( //uses bar chart, able to change options such as configuration choices, like adjusting the legend and enabling responsiveness 
        <>
            <Bar options={options} data={data} /> 
        </>
    )
}

export default App;

