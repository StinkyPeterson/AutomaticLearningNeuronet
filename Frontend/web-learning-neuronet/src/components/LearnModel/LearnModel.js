import './LearnModel.scss';
import {Button, CheckBox, FileUploader, NumberBox, SelectBox} from 'devextreme-react';
import {useState, useEffect} from 'react';
import io from "socket.io-client";
import {Diagram} from '../Diagram/Diagram';

const ENDPOINT = process.env.REACT_APP_BACKEND;
const socket = io(ENDPOINT, {
    timeout: 600000,
    pingTimeout: 600000,
    reconnectionAttempts: 20,
    reconnectionDelay: 5000,
});

export function LearnModel() {
    const [isConnect, setIsConnect] = useState(false)
    const [sid, setSid] = useState(null)
    const [dataModelsLearning, setDataModelsLearning] = useState(null)
    const [modelLearning, setModelLearning] = useState(null)
    const [eraCount, setEraCount] = useState(1)
    const [validationPercent, setValidationPercent] = useState(0.1)
    const [partitionLevel, setPartitionLevel] = useState(1)
    const [isAutomaticStop, setIsAutomaticStop] = useState(false)
    const [isDisabledButton, setIsDisabledButton] = useState(true)
    const [isModelLearning, setIsModelLearning] = useState(false)

    const [epochData, setEpochData] = useState([]);
    const [modelNumber, setModelNumber] = useState(0)

    const [isDatasetLoaded, setIsDatasetLoaded] = useState(false)
    const [isModelEndLearning, setIsModelEndLearning] = useState(false)


    useEffect(() => {
        socket.emit("get_models")
        socket.on("connect", (data) => {
            console.log("Connected to server");
            console.log(socket);
            setIsConnect(true)
            setSid(socket.id) 
            if(socket.recovered){
                console.log('соединение восстановленно')
            }else{
                console.log('новое соединение')
            }
            // setTimeout(() => {
            //     // close the low-level connection and trigger a reconnection
            //     socket.io.engine.close();
            // }, Math.random() * 5000 + 1000);
        });
        socket.on("reconnect", (data) => {
            console.log('recconect')
        })
        socket.on("response", (data) => {
            console.log(data)
        })
        socket.on("disconnect", (reason, details) => {
            console.log("Disconnected from server");
            console.log(socket)
            console.log(reason);
            console.log(details);
            console.log(details?.description);
            console.log(details?.context);
            setIsConnect(false)
            setSid(null)
        });
        socket.on("send_models", (data) => {
            setDataModelsLearning(data);
        });
        socket.on("end_training", () => {
            console.log("Модель завершила обучение!")
            setIsModelEndLearning(true)
        })
        socket.on('dataset_loaded', () => {
            console.log('Датасет загружен!')
            setIsDatasetLoaded(true)
        })
        socket.on('file_data', (data) => {
            downloadFile(data);
          });
        return () => {
            socket.off("send_models");
            socket.off("connect");
            socket.off("disconnect");
        };
    }, []);

    useEffect(() => {
        if (modelLearning !== null && isDatasetLoaded === true) {
            setIsDisabledButton(false)
        }
    }, [modelLearning, isDatasetLoaded])

    useEffect(() => {
        if(isModelLearning){
            console.log("ПОДПИСКА НА СОБЫТИЕ")
            socket.on("send_diagram", (data) => {
                console.log("Received periodic data:", data);
                console.log(modelNumber)
                setEpochData(prevData => ({
                    ...prevData,
                    [modelNumber]: [...(prevData[modelNumber] || []), data],
                }));
            });
        }
    }, [isModelLearning])

    useEffect(() => {
        if(isModelEndLearning){
            socket.off("send_diagram");
        }
    }, [isModelEndLearning])

    function onModelChangedHandler(e) {
        setModelLearning(e.value)
    }

    function onEraChangedHandler(e) {
        setEraCount(e.value)
    }

    function onPartitionChangedHandler(e) {
        setPartitionLevel(e.value)
    }

    function onValidationChangedHandler(e) {
        setValidationPercent(e.value)
    }

    function onAutomaticStopHandler(e) {
        setIsAutomaticStop(e.value)
    }

    function startEducationModel() {
        const dtoEducation = {
            idModel: modelLearning,
            eraCount: eraCount,
            validationPercent: validationPercent,
            partitionLevel: partitionLevel,
            isAutomaticStop: isAutomaticStop,
            dataPath: "",
            xmlPath: "",
        }
        // console.log("модель: ", dtoEducation)
        socket.emit("start", dtoEducation)
        setIsModelLearning(true)
        setIsModelEndLearning(false)
    }

    function downloadFile(data){
        const byteCharacters = atob(data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'file.pt';
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
      };

    async function handleFileUpload(event){
        console.log(event)
        const file = event.target.files[0];
        if (file) {
          await sendZipFile(file);
        }
      };

    async function sendZipFile(file) {
        const CHUNK_SIZE = 1024 * 1024; // 1 МБ
        const reader = new FileReader();
        reader.onload = async () => {
          const buffer = reader.result;
          let offset = 0;
          while (offset < buffer.byteLength) {
            const chunk = buffer.slice(offset, offset + CHUNK_SIZE);
            offset += CHUNK_SIZE;
            socket.emit('chunk', chunk);
            await new Promise((resolve) => setTimeout(resolve, 100)); // Пауза между чанками
          }
          await new Promise((resolve) => setTimeout(resolve, 1000)); // Пауза между чанками
          socket.emit('unpacking_dataset');
        };
        reader.readAsArrayBuffer(file);
      }

      function handleButtonClick (){
        socket.emit('download_file');
      };

      function handleNewTraining(){
          setModelNumber(modelNumber + 1)
          setIsModelLearning(false)
      }

    return (
        <div>
            {
                isConnect && <p>Подключение установлено: {sid}</p>
            }
            {
                !isConnect && <p>Подключение не установлено</p>
            }
            {!isModelLearning &&
                <>
                    <h1 className="h1-model" >Настройка модели</h1>
                    <div className='model-setting-container'>
                        <div className='select-model-learning'>
                            <SelectBox
                                label='Модель обучения'
                                labelMode="outside"
                                stylingMode="outlined"
                                dataSource={dataModelsLearning}
                                displayExpr="value"
                                valueExpr="id"
                                onValueChanged={onModelChangedHandler}
                            />
                        </div>
                        <div className='setting-small-param'>
                            <div className='setting-col'>
                                <NumberBox
                                    label='Количество эпох'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    format="#"
                                    defaultValue={eraCount}
                                    onValueChanged={onEraChangedHandler}
                                    min={1}
                                />
                                <NumberBox
                                    label='Процент валидации'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    format="#0%"
                                    defaultValue={validationPercent}
                                    onValueChanged={onValidationChangedHandler}
                                    min={0.1}
                                    max={1}
                                />
                                <NumberBox
                                    label='Ширина (px)'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    defaultValue={224}
                                />
                            </div>
                            <div className='setting-col'>
                                <NumberBox
                                    label='Размер партиции'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    format="#"
                                    defaultValue={partitionLevel}
                                    onValueChanged={onPartitionChangedHandler}
                                    min={1}
                                />
                                <NumberBox
                                    label='Интенсивность'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    defaultValue={0.01}
                                />
                                <NumberBox
                                    label='Высота (px)'
                                    labelMode="outside"
                                    stylingMode="outlined"
                                    defaultValue={224}
                                />



                            </div>
                            <div className='setting-col input-file-col'>
                                {/*<input*/}
                                {/*    type="file"*/}
                                {/*    onChange={handleFileUpload}*/}
                                {/*    className='input-file'*/}
                                {/*/>*/}
                                <form method="post" encType="multipart/form-data">
                                    <label class="input-file">
                                        <input type="file" name="file" onChange={handleFileUpload} />
                                            <span class="input-file-btn">Выберите файл</span>
                                    </label>
                                </form>
                                <div className='select-setting'>
                                    <p>Автоматическая остановка</p>
                                    <CheckBox
                                        defaultValue={isAutomaticStop}
                                        onValueChanged={onAutomaticStopHandler}
                                    />
                                </div>
                            </div>
                        </div>
                        <Button
                            text="Начать обучение"
                            type="success"
                            className='start-learning'
                            disabled={isDisabledButton}
                            onClick={startEducationModel}
                        />
                    </div>
                </>
            }
            {
                isModelEndLearning &&
                <>
                    <Button text ="Скачать файл модели" className="btn-training" onClick={handleButtonClick} />
                    <Button text="Начать новое обучение модели" className="btn-training" onClick={handleNewTraining}/>
                </>
            }
            {Object.keys(epochData).reverse().map(modelName => (
                <div key={modelName}>
                    <Diagram key={modelName} data={epochData[modelName]} />
                </div>
            ))}
        </div>
    );
}