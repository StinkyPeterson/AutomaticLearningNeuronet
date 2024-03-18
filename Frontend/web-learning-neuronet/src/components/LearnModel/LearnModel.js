import './LearnModel.scss';
import {Button, CheckBox, FileUploader, NumberBox, SelectBox} from 'devextreme-react';
import {useState, useEffect} from 'react';
import io from "socket.io-client";
import {Diagram} from '../Diagram/Diagram';

const ENDPOINT = "http://localhost:8765";
const socket = io(ENDPOINT, {
    timeout: 600000,
    pingTimeout: 600000,
    reconnectionAttempts: 10,
    reconnectionDelay: 1000,
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

    const [isDatasetLoaded, setIsDatasetLoaded] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [isModelEndLearning, setIsModelEndLearning] = useState(false)


    useEffect(() => {
        socket.emit("get_models")
        socket.on("connect", (data) => {
            console.log("Connected to server");
            setIsConnect(true)
            setSid(socket.id) 
            if(socket.recovered){
                console.log('соединение восстановленно')
            }else{
                console.log('новое соединение')
            }
        });
        socket.on("response", (data) => {
            console.log(data)
        })
        socket.on("disconnect", (reason, details) => {
            console.log("Disconnected from server");
            console.log(reason);
            //console.log(details?.message);
            console.log(details?.description);
            console.log(details?.context);
            setIsConnect(false)
            setSid(null)
        });
        socket.on("send_models", (data) => {
            console.log("Received models:", data);
            setDataModelsLearning(data);
        });
        socket.on("send_diagram", (data) => {
            console.log("Received periodic data:", data);
            setEpochData(prevData => [...prevData, data]);
        });
        socket.on("end_training", () => {
            console.log("Модель завершила обучение!")
            setIsModelEndLearning(true)
        })
        socket.on('dataset_loaded', () => {
            console.log('Датасет загружен!')
            setIsDatasetLoaded(true)
            setIsLoading(false)
        })
        socket.on('file_data', (data) => {
            downloadFile(data);
          });
        return () => {
            socket.off("send_models");
        };
    }, []);

    useEffect(() => {
        if (modelLearning !== null && isDatasetLoaded === true) {
            setIsDisabledButton(false)
        }
    }, [modelLearning, isDatasetLoaded])

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
        console.log(e)
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
        setIsLoading(true)
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
                    <h1>Настройка модели</h1>
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
                                <div className='select-setting'>
                                    <p>Автоматическая остановка</p>
                                    <CheckBox
                                        defaultValue={isAutomaticStop}
                                        onValueChanged={onAutomaticStopHandler}
                                    />
                                </div>

                            </div>
                            <div className='setting-col'>
                                <input
                                    type="file"
                                    onChange={handleFileUpload}
                                />
                                {/* <FileUploader
                                    width={400}
                                    labelText='Форма загрузки XML'
                                    selectButtonText="Выберите файл .xml"
                                    uploadMode="useForm"
                                    allowedFileExtensions={[".xml"]}
                                /> */}
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
                isModelLearning &&
                <>
                    <Diagram data={epochData}/>
                </>
            }
            {
                isModelEndLearning && 
                <Button text ="Скачать файл модели" onClick={handleButtonClick} />
            }

        </div>
    );
}