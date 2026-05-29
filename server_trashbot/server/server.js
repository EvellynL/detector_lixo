const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
app.use(cors());

app.use(express.json());

const path = require('path');
// const { act } = require('react');
// const { stat } = require('fs');

app.use(
    '/uploads',
    express.static(path.join(__dirname, 'uploads'))
);

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');

    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }

});

const upload = multer({ storage });

let raspberryStatus ={
    conectado: false,
    ultimoPing: null,
}

let trashcounts = {
    PAPER: 0,
    METAL: 0,
    GLASS: 0,
}

let latestImage = '';

let latestImageTime = '--.--.--';

let systemStatus = 'Aguardando conexão...';

let activitylog= [];

function addLog(message){
    const time = new Date().toLocaleString();

    activitylog.unshift(
        `[${time}] ${message}`
    );

    if(activitylog.length > 10){
        activitylog.pop();
    }
}

app.post('/api/systemStatus', (req, res) => {
    const status = req.body.status;

    systemStatus = status;

    addLog(status);

    console.log('status:', status);

    res.json({
        success: true
    });
});

app.post(
    '/api/classification',
    upload.single('image'),

    (req, res) => {

        try {

            console.log(req.file);
            console.log(req.body);

            if (!req.file) {

                return res.status(400).json({
                    error: 'Imagem não enviada'
                });

            }

            const classe = req.body.class;

            if (trashcounts[classe] !== undefined) {

                trashcounts[classe]++;

            }

            latestImage =
                    `/uploads/${req.file.filename}`;

            latestImageTime = new Date().toLocaleString();

            res.json({
                success: true
            });

        } catch (error) {

            console.log(error);

            res.status(500).json({
                error: 'Erro no upload'
            });

        }

    }
);

app.post('/api/connect', (req, res) => {

     if (!raspberryStatus.conectado) {

        console.log("Raspberry conectado!");

    }

    raspberryStatus.conectado = true;
    raspberryStatus.ultimoPing = new Date();

    res.json({
        sucess: true,
        message: 'Conectado ao servidor'

    });
});

app.get('/api/status', (req, res) => {
    res.json({
        conectado: raspberryStatus.conectado,
        contadores: trashcounts,
        latestImage: latestImage,
        latestImageTime: latestImageTime,
        systemStatus: systemStatus,
        logs: activitylog,
    });
});

setInterval(() => {
    if(!raspberryStatus.conectado) return;

    const now = new Date();
    const diff = (now - raspberryStatus.ultimoPing)/1000;

    if(diff > 50  && raspberryStatus.conectado){
        raspberryStatus.conectado = false;
        console.log('Raspberry Pi desconectado por timeout');
    }
}, 1000);

app.listen(3000, "0.0.0.0", () => {
    console.log("Servidor rodando na porta 3000");
});

