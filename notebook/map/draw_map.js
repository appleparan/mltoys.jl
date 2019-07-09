import * from 'jsdom';
import * from 'leafle't;
import DataFrame from 'dataframe-js';

const station_csv = "/mnt/post/station/station_stats.csv"

function getData(df, ycol) {
    const df_ycol = df.filter(row => row.get('colname') == ycol);
    // text_arr = df_ycol.map(row => (row.get('name') + '<br> RSR: ' + row.get('RSR') + '<br> corr_24h: ' + row.get('corr_24h'))).toArray();
    const name_arr = df_ycol.select('name').toArray();
    const RSR_arr = df_ycol.select('RSR').toArray();
    const corr_24h_arr = df_ycol.select('corr_24h').toArray();
    
    const text_arr = name_arr.map((e, i) => {
        return e + '<br> RSR: ' + RSR_arr[i] + '<br> corr_24h: ' + corr_24h_arr[i]
    });
                 
    const data = [{
        type: 'scattergeo',
        mode: 'markers+text',
        text: text_arr,
        lon: df_ycol.select('lon').toArray(),
        lat: df_ycol.select('lat').toArray(),
        marker: {
            size: 7,
        },
        name: ycol,
    }];
    
    return data;
}

function getLayout(ycol) {
    var layout = {
        title: ycol,
        font: {
            family: 'NanumGothic, san-serif',
            size: 6
        },
        titlefont: {
            size: 16
        },
        geo: {
            scope: 'asia',
            resolution: 100,
            lonaxis: {
                'range': [126.5, 127.5]
            },
            lataxis: {
                'range': [37.5, 37.7]
            },
            showrivers: true,
            rivercolor: '#fff',
            showlakes: true,
            lakecolor: '#fff',
            showland: true,
            landcolor: '#EAEAAE',
            countrycolor: '#d3d3d3',
            countrywidth: 1.5,
            subunitcolor: '#d3d3d3'
        }
    };
    
    return layout;
}

DataFrame.fromCSV(station_csv)
    .then(df => {
        const ycol = "PM10";
        const data = getData(df, ycol);
        const layout = getLayout(ycol);
        var map = new L.Map("map", {center: [37.532600, 127.024612], zoom: 12})
        .addLayer(new L.TileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"));
        // var myPlot = Plotly.createPlot(data, layout)
        // $$html$$ = myPlot.render();
    })
    .catch(error => console.error(error));

