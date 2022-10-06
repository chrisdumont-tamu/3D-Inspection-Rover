import './App.css';
import { useState, useEffect } from 'react';
import Histogram from "react-chart-histogram";

function App() {
    const [tomatoes, setTomatoes] = useState([]);
    const labels = ['Green','Breaker','Turning','Pink','Light Red','Red','Deffective'];
    const options = {fillColor: '#FFFFFF', strokeColor: '#0000FF'};
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
        <table>
            <thead>
            <th> Results </th>
                <tr>
                    <th> Classification </th>
                    <th> Volume </th>
                </tr>
            </thead>

            {tomatoes.length > 0 && (
                <tbody>
                    {tomatoes.map(tomato => (
                        <tr key={tomato.id}>
                            <td>{tomato.Stage}</td>
                            <td>{tomato.Aisle}</td>
                        </tr>
                    ))}
                </tbody>
            )}
        </table>

        <Histogram
            xLabels = {labels}
            yValues = {[tomatoes.length, 6, 7, 8, 10, 12, 14]}
            width = '400'
            height = '200'
            options = {options}

        />
        </>
    )
}

export default App;