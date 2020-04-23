const createError = require('http-errors');
const express = require('express');
// const session = require('express-session');
// const pgSession = require('connect-pg-simple')(session);
// const passport = require('passport');
// require('./config/pass')(passport);
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');
const multer = require('multer');
const indexRouter = require('./routes/index');
const dashRouter = require('./routes/dashboard');
// const loginRouter = require('./routes/login');
// const logoutRouter = require('./routes/logout');
// const registerRouter = require('./routes/register');
// const dbPool = require('./database');
const cors = require('cors');


const app = express();


// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'hbs');

// var multer = require('multer');

// app.use(multer({dest:'./uploads/'}).single('singleInputFileName'));
app.use(cors());
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
// app.use(session({
//   store: new pgSession({
//     pool: dbPool,
//     tableName: 'session',
//   }),
//   secret: "mysecret",
//   resave: false,
//   saveUninitialized: true,
//   cookie: { maxAge: 30 * 24 * 60 * 60 * 1000 } // 30 days
// }));
// app.use(passport.initialize());
// app.use(passport.session());

app.use('/', indexRouter);
// app.use('/login', loginRouter);
// app.use('/logout', logoutRouter);
// app.use('/register', registerRouter);
app.use('/dashboard', dashRouter);


app.get('/analysis', async (req, res) => {
  // const username = req.user.username;

  res.render('getall', {
    title: 'Analysis Of The Models',
    // username
  });
});

app.get('/dashboard/lstm', async (req, res) => {
  // const username = req.user.username;

  res.render('lstm', {
    title: 'Long Short Term Memory',
    // username
  });
});

app.get('/dashboard/cnnlstm', async (req, res) => {
  // const username = req.user.username;

  res.render('cnnlstm', {
    title: 'Convolutional Long Short Term Memory',
    // username
  });
});

app.get('/dashboard/gru', async (req, res) => {
  // const username = req.user.username;

  res.render('gru', {
    title: 'Gated Recurrent Unit',
    // username
  });
});

app.get('/dashboard/cnngru', async (req, res) => {
  // const username = req.user.username;

  res.render('cnngru', {
    title: 'Convolutional Gated Recurrent Unit',
    // username
  });
});

app.get('/dashboard/cnn', async (req, res) => {
  // const username = req.user.username;

  res.render('cnn', {
    title: 'Convolutional Neural Network',
    // username
  });
});


app.get("/getExcelData", (req, res) => {
  var workbook = XLSX.readFile(__dirname + "/views/excel/data.xlsx");
  var sheet_name_list = workbook.SheetNames;
  var xlData = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[0]]);
  
  my_json = {   
    'time_1'  : time_1,
    'stock_1' : stock_1,
    'time_15': time_15,
    'CNNLSTM'   : list_CL,
    'CNNGRU'    : list_CG,
    'GRU'       : list_G,
    'LSTM'      : list_L,
    'CNN'       : list_C
}
  res.json(my_json);
});


// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
