import express from "express";
import bodyParser from "body-parser";
import pg from "pg";
import bcrypt from "bcrypt";
import flash from "connect-flash";
import session from "express-session";
import path from "path";
import fs from "fs";
import multer from "multer";
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 4000;
const saltRounds = 15;

const db = new pg.Client({
    user: "postgres",
    host: "localhost",
    database: "prostateXAI",
    password: "xbsb435&",
    port: 5432,
  });
  db.connect();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, 'mri images/'); // Directory to save uploaded files
    },
    filename: (req, file, cb) => {
      cb(null, Date.now() + '-' + file.originalname); // Unique filename
    }
  });
const upload = multer({ storage: storage });
// Ensure the "mri images" directory exists
const uploadsDir = path.join(__dirname, 'mri images');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}
  
app.use(express.static("public"));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(session({
    secret: "konkolukuulu",
    resave: false,
    saveUninitialized: true,
}));
app.use(flash());

app.use((req, res, next) => {
    res.locals.messages = req.flash();
    next();
});

// Route to handle the homepage
app.get("/", (req, res) => {
    res.render("index.html");
  });

// Route to handle the create signup page
app.get("/create", (req, res) => {
    res.render("signup.ejs", { messages: req.flash() });
  });
// Route to handle the signup page
app.post("/signup", async(req, res) => {
    const { fname, lname, email, phone, password, cpassword } = req.body;
    console.log(fname);
    console.log(lname);
    console.log(phone);
    console.log(password);
    console.log(email);

    if (password !== cpassword) {
        req.flash("error",`Passwords do not match. Make sure passwords match`);
        res.redirect("/create");
    }
    
    try {
        const checkResult = await db.query("SELECT * FROM admins WHERE email = $1", [email]);
        if (checkResult.rows.length > 0) {
            req.flash("error",`User with ${email} already. Try logging in.`);
            res.redirect("/create");
        } else {
            // Password hashing
            bcrypt.hash(password, saltRounds, async(err, hash) => {
                if (err) {
                    console.log("Error hashing password", err);
                } else {
                    const result = await db.query(
                        "INSERT INTO admins (first_name, last_name, email, phone_number, password) VALUES ($1, $2, $3, $4, $5)",
                        [fname, lname, email, phone, hash]
                    );
                    console.log(result);
                    res.render("upload.ejs"); 
                } 
            });
            
        } 
    } catch (error) {
       console.log(error); 
    }
  });
// Route to handle the welcome login page
app.get("/signin", (req, res) => {
    res.render("login.ejs", { messages: req.flash() });
  });
// Route to handle the login page
app.post("/login", async(req, res) => {
    const {loginEmail, loginPassword} = req.body; 
    try {
       const result = await db.query("SELECT * FROM admins WHERE email = $1", [loginEmail]);
       if(result.rows.length > 0) {
            const user = result.rows[0];
            const storedHashedPassword = user.password;
            bcrypt.compare(loginPassword, storedHashedPassword, (err, result) => {
                if (err) {
                    console.log("Error comparing passwords:", err);
                } else {
                  if (result) {
                    res.render("upload.ejs");
                  } else {
                    console.log(result);
                    req.flash("error","Incorrect password");
                    res.redirect("/signin");
                  }  
                }
            })
       }  else {
            req.flash("error","User not found");
            res.redirect("/signin");
       }
    } catch (error) {
       console.log(error); 
    }
  });

// Route for the get request of scan page
app.get('/upload', (req, res) => {
    res.render('upload.ejs', { messages: req.flash() });
  });
// Route to handle the scan page
app.post("/scan", upload.array('images', 3), (req, res) => {
    const {patientName, patientID, gender, birthDate} = req.body;
    console.log(patientName, patientID, gender, birthDate);
    console.log('Flash messages:', req.flash());
    if (!req.files || req.files.length === 0 || req.files.length !== 3) {
        req.flash("error", "No files uploaded. Make sure to upload 3 files");
        return res.redirect('/upload');
      }
    else if(!patientName || !patientID || !gender || !birthDate) {
        req.flash('error', 'Make sure all fields are entered correctly.');
        return res.redirect('/upload');
    }
    else {
        res.status(200).json({success: "Will redirect you to result page soon"});
    }
  });

app.listen(port, () => {
    console.log(`API is running at http://localhost:${port}`);
  });