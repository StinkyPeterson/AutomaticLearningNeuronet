export async function sendSetting(settings){
    const request = await fetch("path_api", {
        method: "POST",
        body: JSON.stringify(settings)
    })
}

export async function sendDataSet(dataSet){
    const request = await fetch("path_api", {
        method: "POST",
        body: dataSet
    })
}