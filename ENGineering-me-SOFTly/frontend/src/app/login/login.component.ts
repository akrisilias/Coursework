import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {

  username: string;
  password: string;

  error: string = '';

  constructor(private data: DataService, private router: Router) { }

  ngOnInit() {
  }

  postClick() {
    this.data.postLogin(this.username, this.password)
      .subscribe( data => {
        console.log(data);

        if ((data as any).token == "Invalid input") {
          this.error = "Your username and password don't match. Please try again.";
        }
        else {
          this.error = "";
          this.data.auth = (data as any).token;
          this.data.user = this.username;
          sessionStorage.setItem("auth", this.data.auth);
          sessionStorage.setItem("user", this.data.user);
          this.router.navigate(['']);
        }
      });
  }

}
