import './LearnModel.scss';
import { Button, CheckBox, FileUploader, NumberBox, SelectBox } from 'devextreme-react';
import { useState, useEffect } from 'react';
import socketIOClient from 'socket.io-client';
import { Diagram } from '../Diagram/Diagram';

const ENDPOINT = "http://localhost:5000";
const socket = socketIOClient(ENDPOINT, {
    timeout: 600000,
    pingTimeout: 600000
});

export function LearnModel(){
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

    const [selectedFile, setSelectedFile] = useState(null);

    useEffect(() => {
          const savedSid = localStorage.getItem("sid");
          if (savedSid) {
              socket.io.opts.query = { sid: savedSid }; // Передача sid в качестве параметра при повторном подключении
              socket.connect();
          }
  

        socket.on("connect", (data) => {
            console.log("Connected to server");
            setIsConnect(true)
            setSid(socket.id)
            localStorage.setItem("sid", socket.id);
        });

        socket.on("disconnect", (reason, details) => {
            console.log("Disconnected from server");
            console.log(reason);
            //console.log(details?.message);
            console.log(details?.description);
            console.log(details?.context);
            setIsConnect(false)
        });

        socket.on("response", (data) => {
            console.log("Received response:", data);
        });
        socket.on("send_models", (data) => {
          setDataModelsLearning(data);
      });
      socket.on("send_diagram", (data) => {
        console.log("Received periodic data:", data);
          setEpochData(prevData => [...prevData, data]);
        //setCount(data);
    });
      socket.on("end_training", () => {
          console.log("Модель завершила обучение!")
      })
        return () => {
            socket.off("send_models");
        };
    }, []);

    useEffect(() => {
      socket.emit("get_models")
    }, [])
  
    useEffect(() => {
      if(modelLearning !== null){
        setIsDisabledButton(false)
      }
    }, [modelLearning, eraCount, validationPercent, partitionLevel])
  
    function onModelChangedHandler(e){
      setModelLearning(e.value)
    }
  
    function onEraChangedHandler(e){
      setEraCount(e.value)
    }
  
    function onPartitionChangedHandler(e){
      setPartitionLevel(e.value)
    }
  
    function onValidationChangedHandler(e){
      console.log(e)
      setValidationPercent(e.value)
    }
  
    function onAutomaticStopHandler(e){
      setIsAutomaticStop(e.value)
    }
    
    function startEducationModel(){
      const dtoEducation = {
        idModel: modelLearning,
        eraCount: eraCount,
        validationPercent: validationPercent,
        partitionLevel: partitionLevel,
        isAutomaticStop: isAutomaticStop,
        dataPath: "",
        xmlPath: "",
      }
      console.log("модель: ", dtoEducation)
      socket.emit("start", dtoEducation)
      //socket.emit("start_periodic_data")
      setIsModelLearning(true)
    }

    const handleFileChange = (event) => {
      setSelectedFile(event.target.files[0]);
      console.log(event)
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
          <div>
            <input type="file" onChange={handleFileChange} />
          </div>
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
                  <FileUploader 
                  width={400}
                  labelText='Форма загрузки XML'
                  selectButtonText="Выберите файл .xml" 
                  uploadMode="useForm" 
                  allowedFileExtensions={[".xml"]}
                />
              </div>
            </div>
            <Button 
            text= "Начать обучение"
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
                <Diagram data = {epochData}/>
            </>
        }

      </div>
    );
}