var express = require('express');
var router = express.Router();


/* GET home page. */
router.get('/', async (req, res) => {

  const username = req.user.username;


  res.render('dashboard', {
    title: 'Welcome '+username+"!!",
    username
  });
});


module.exports = router;
