import './App.css'
import logoReciclagem from './assets/reciclagem.png'
import ImgDeteccao from './assets/deteccao.png'
import { useEffect, useState } from 'react'

function App() {

  const [status, setStatus] = useState('Conectando...');

  const [latestImage, setLatestImage] = useState('');

  const[latestImageTime, setLatestImageTime] = useState('--.--.--');

  const [systemStatus, setSystemStatus] = useState('');

  const [logs, setLogs] = useState([]);

  const[counts, setCounts] = useState({
    PAPER: 0,
    METAL: 0,
    GLASS: 0,
  });

  async function getStatus(){
    try{
      const response = await fetch('http://localhost:3000/api/status');
      
      const data = await response.json();

      if(data.conectado){
        setStatus('Conectado');
      }else{
        setStatus('Aguardando conexão...');
      }
      setCounts(data.contadores);
      setLatestImage(data.latestImage);
      setLatestImageTime(data.latestImageTime);
      setLogs(data.logs);

      console.log("JSON COMPLETO:", data);
      console.log("SYSTEM STATUS RECEBIDO:", data.systemStatus);

      setSystemStatus(data.systemStatus);
    }
    catch {
      setStatus('servidor offline');
    }
  }

  useEffect(() => {
    getStatus();
    const interval = setInterval(getStatus, 1000);
    return () => clearInterval(interval);
  },[]);

  return (
    <>
      <header>
          <div className="logo">
            <img src={logoReciclagem} alt="Logo" />
            <h1>Trash<span>Bot</span></h1>
          </div>
      </header>

      <main className="dashboard-grid">
        
        {/* Bloco 1: Status do Sistema */}
        <section className="card status-card">
          <h3>SYSTEM STATUS</h3>
          <h2>STATUS: {systemStatus}</h2>
          <p>Raspberry Pi:   <span
            style={{

              color:
                status === 'Conectado'
                  ? 'limegreen'
                  : status === 'Aguardando conexão...'
                  ? 'red'
                  : 'gray',

              fontWeight: 'bold'

            }}
          >
            {status}
          </span></p>
          <p>Última Foto: {latestImageTime}</p>
        </section>

        {/* Bloco 2: Totais de Detecção */}
        <section className="card totals-card">
          <h3>DETECTION TOTALS</h3>
          <h2>ITENS DETECTADOS</h2>
          {/* Aqui dentro você colocará seus gráficos/barras depois */}
          <div className="classes-container">
            <div id='paper' className="card-trash">Papel: {counts.PAPER}</div>
            <div id='metal' className="card-trash">Metal: {counts.METAL}</div>
            <div id='glass' className="card-trash">Vidro: {counts.GLASS}</div>
          </div>
        </section>

        {/* Bloco 3: Feed de Imagem ao Vivo */}
        <section className="card feed-card">
          <h3>LIVE FEED</h3>
          <div className="video-placeholder">

              <img
                key={latestImage}
                src={`http://localhost:3000${latestImage}?t=${Date.now()}`}
                alt="Feed da Esteira"
                style={{
                  width: '100%',
                  height: '300px',
                  objectFit: 'cover',
                  border: '4px solid red'
                }}
              />

            </div>
        </section>

        {/* Bloco 4: Log de Atividade */}
        <section className="card log-card">
          <h3>LOG DE ATIVIDADE</h3>
          <div className="log-container">
              {
              logs.map((log, index) => (

                <p key={index}>
                  {log}
                </p>

              ))
            }
          </div>
        </section>

      </main>

      <footer>
        <p>© 2026 <strong>TRASHBOT</strong> - Sistema Autônomo de Triagem com IA</p>
      </footer>
    </>
  )
}

export default App
