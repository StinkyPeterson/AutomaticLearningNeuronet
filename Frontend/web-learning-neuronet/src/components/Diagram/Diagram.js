import React, { useState } from "react";
import {Legend, Line, LineChart, XAxis, YAxis} from "recharts";


export function Diagram({data}){
    console.log(data)
    return(
        <LineChart width={500} height={500} data={data}>
            <XAxis dataKey="epoch"/>
            <YAxis />
            <Line type="monotone" dataKey="trainLossY" stroke="#8884d8" />
            <Line type="monotone" dataKey="trainDiceY" stroke="#82ca9d" />
            <Line type="monotone" dataKey="valLossY" stroke="#c93838" />
            <Line type="monotone" dataKey="valDiceY" stroke="#1fbdcf" />
            <Legend />
        </LineChart>
    )
}