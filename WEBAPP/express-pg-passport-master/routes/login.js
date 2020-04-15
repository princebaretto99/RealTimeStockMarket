const express = require('express');
const passport = require('passport')
const router = express.Router();

router.get('/', function(req, res) {
  if (req.user) {
    res.redirect('dashboard')
  } else {
    res.render('login');
  }
});

router.post('/', passport.authenticate('local', { 
  successRedirect: '/dashboard', 
  failureRedirect: '/login'
}));

module.exports = router;