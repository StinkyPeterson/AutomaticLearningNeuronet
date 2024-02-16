import 'devextreme/dist/css/dx.light.css';
import logo from './logo.svg';
import './App.css';
import { Button, CheckBox, FileUploader, NumberBox, SelectBox } from 'devextreme-react';
import { useState } from 'react';
import { dataModels } from './domain/data';

function App() {
  const [modelLearning, setModelLearning] = useState(null)
  const [eraCount, setEraCount] = useState(1)
  const [validationPercent, setValidationPercent] = useState(0.1)
  const [partitionLevel, setPartitionLevel] = useState(1)
  const [isAutomaticStop, setIsAutomaticStop] = useState(false)

  const [isDisabledButton, setIsDisabledButton] = useState(true)

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
    setValidationPercent(e.value)
  }

  function onAutomaticStopHandler(e){
    setIsAutomaticStop(e.value)
  }

  return (
    <div className="App">
      <h1>Настройка модели</h1>
      <h3>Модель обучения</h3>
      <SelectBox  
        dataSource={dataModels}
        displayExpr="value"
        valueExpr="id"
        onValueChanged={onModelChangedHandler}
      />
      <h3>Количество эпох</h3>
      <NumberBox 
        format="#"
        defaultValue={eraCount}
        onValueChanged={onEraChangedHandler}
        min={1}
      />
      <h3>Процент валидации</h3>
      <NumberBox 
        format="#0%"
        defaultValue={validationPercent}
        onValueChanged={onValidationChangedHandler}
        min={0.1}
        max={1}
      />
      <h3>Размер партиции</h3>
      <NumberBox 
        format="#"
        defaultValue={partitionLevel}
        onValueChanged={onPartitionChangedHandler}
        min={1}
      />
      <h3>Автоматическая остановка</h3>
      <CheckBox 
        defaultValue={isAutomaticStop}
        onValueChanged={onAutomaticStopHandler}
      />
      <h3>Форма загрузки файла</h3>
      <FileUploader 
        selectButtonText="Выберите файл .zip, .rar" 
        uploadMode="useForm" 
        allowedFileExtensions={[".zip", ".rar"]}
      />
      <h3>Форма загрузки</h3>
      <FileUploader 
        selectButtonText="Выберите файл .xml" 
        uploadMode="useForm" 
        allowedFileExtensions={[".xml"]}
      />
      <Button 
        text= "Начать обучение"
        type="success"
        disabled={isDisabledButton}
      />
    </div>
  );
}

export default App;
