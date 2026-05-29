import { useState } from 'react'
import './App.css'
import logoReciclagem from './assets/reciclagem.png'
import ImgDeteccao from './assets/deteccao.png'

function App() {
  const [count, setCount] = useState(0)

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
          <h2>STATUS: AGUARDANDO ITEM</h2>
          <p>Raspberry Pi: <span>Conectado</span></p>
          <p>Última Foto: 14:05:31</p>
        </section>

        {/* Bloco 2: Totais de Detecção */}
        <section className="card totals-card">
          <h3>DETECTION TOTALS</h3>
          <h2>ITENS DETECTADOS</h2>
          {/* Aqui dentro você colocará seus gráficos/barras depois */}
          <div className="classes-container">
            <div id='paper' className="card-trash">Papel: 20</div>
            <div id='metal' className="card-trash">Metal: 30</div>
            <div id='glass' className="card-trash">Vidro: 10</div>
          </div>
        </section>

        {/* Bloco 3: Feed de Imagem ao Vivo */}
        <section className="card feed-card">
          <h3>LIVE FEED</h3>
          <div className="video-placeholder">
            <img src={ImgDeteccao} alt="Feed da Esteira" />
          </div>
        </section>

        {/* Bloco 4: Log de Atividade */}
        <section className="card log-card">
          <h3>LOG DE ATIVIDADE</h3>
          <div className="log-container">
            <p>[14:05:31] - Papel Detectado</p>
            <p>[14:05:28] - Realizando Triagem - Metal</p>
            <p>[14:05:25] - Item Triado</p>
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
